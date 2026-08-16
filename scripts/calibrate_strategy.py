"""
scripts/calibrate_strategy.py

Standalone calibration/backtest tool for `TrendEmaStrategy`, driven by
REAL historical 1-minute candles pulled directly from the dYdX v4 Indexer
(never synthetic/mocked data). Answers three questions:

    1. How many signals does the strategy actually generate over N real
       days, and how many get filtered out by each risk control
       (confirmation window, volatility filter, cooldown)?
    2. If those signals had been traded (single-position, walk-forward,
       no lookahead), what would win rate / profit factor / max drawdown
       / total return have looked like?
    3. Given the REAL ATR% distribution and a small grid of EMA period
       pairs, which settings would have performed best over this window?

This is a signal-generation and PnL-shape calibration tool, not a
production backtester: trades are sized at a fixed 1x notional (no
leverage, no fees beyond a flat taker-fee assumption both ways) so the
numbers reflect strategy quality, not position-sizing choices — those are
RiskManager's job, not TrendEmaStrategy's.

Usage:
    python3 scripts/calibrate_strategy.py --symbol ETH-USD --days 5
    python3 scripts/calibrate_strategy.py --symbol ETH-USD --days 7 --cache-file /tmp/eth_candles.csv

Author: Senior Python/Crypto Backend Engineer
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from exchange.dydx_v4_adapter import _with_retries
from exchange.indexer_http import indexer_get, normalize_and_validate_indexer_url
from strategies.trend_ema import TrendEmaStrategy
from strategies.trend_pullback import TrendPullbackStrategy

logger = logging.getLogger("calibrate_strategy")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

TAKER_FEE_PCT = 0.05  # 0.05% per side, matches dYdX v4 / PaperExchange default
CANDLES_PAGE_LIMIT = 100  # conservative per-request page size for the Indexer
INTER_PAGE_DELAY_SECONDS = 0.3  # proactive pacing to avoid rate limits


# --------------------------------------------------------------------------- #
# Step 1: Fetch real historical candles from the Indexer, paginated
# --------------------------------------------------------------------------- #

async def fetch_historical_candles(
    symbol: str,
    days: float,
    indexer_url: str,
    resolution: str = "1MIN",
    page_limit: int = CANDLES_PAGE_LIMIT,
) -> pd.DataFrame:
    """
    Fetch `days` worth of real 1-minute candles for `symbol` from the
    dYdX v4 Indexer, walking backward in time via `toISO` pagination since
    a single request only returns up to `page_limit` candles.

    Rate-limit handling: each page fetch goes through `_with_retries`
    (exponential backoff + jitter on any failure, including HTTP 429), and
    a small fixed delay is added between successful page fetches to avoid
    tripping rate limits in the first place rather than only reacting
    after the fact.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    all_rows: List[dict] = []
    cursor_to_iso: Optional[str] = None  # None on first request = "now"
    page_num = 0
    seen_earliest: Optional[pd.Timestamp] = None

    logger.info(
        "Fetching %s %s candles from %s to %s (target ~%.0f candles for %.1f days)...",
        symbol, resolution, start_time.isoformat(), end_time.isoformat(),
        days * 24 * 60, days,
    )

    while True:
        page_num += 1
        params = {"resolution": resolution, "limit": page_limit}
        if cursor_to_iso is not None:
            params["toISO"] = cursor_to_iso

        async def _fetch(p=params) -> dict:
            return await indexer_get(
                indexer_url, f"/v4/candles/perpetualMarkets/{symbol}", params=p
            )

        try:
            response = await _with_retries(
                _fetch, op_name=f"fetch_candles page {page_num}", max_attempts=5,
                base_delay_seconds=1.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Giving up fetching page %d after retries: %s", page_num, exc)
            break

        candles = response.get("candles", [])
        if not candles:
            logger.info("Page %d returned no candles — reached start of available history.", page_num)
            break

        all_rows.extend(candles)

        # Indexer returns candles newest-first within a page.
        page_timestamps = [pd.Timestamp(c["startedAt"]) for c in candles]
        earliest_in_page = min(page_timestamps)
        newest_in_page = max(page_timestamps)

        logger.info(
            "Page %d: %d candles, range [%s .. %s], total so far: %d",
            page_num, len(candles), earliest_in_page, newest_in_page, len(all_rows),
        )

        if seen_earliest is not None and earliest_in_page >= seen_earliest:
            # Pagination stalled (identical/overlapping page) — stop to
            # avoid an infinite loop rather than trusting the API forever.
            logger.warning("Pagination did not advance on page %d — stopping.", page_num)
            break
        seen_earliest = earliest_in_page

        if earliest_in_page <= pd.Timestamp(start_time):
            logger.info("Reached target start time (%s) — stopping pagination.", start_time)
            break

        # Walk the cursor back to just before the earliest candle we've
        # seen so far, so the next page continues where this one left off.
        cursor_to_iso = (earliest_in_page - pd.Timedelta(seconds=1)).isoformat()

        await asyncio.sleep(INTER_PAGE_DELAY_SECONDS)

    if not all_rows:
        raise RuntimeError(
            f"No candle data fetched for {symbol} — check the symbol name and "
            f"that DYDX_V4_INDEXER_URL is reachable."
        )

    rows = [
        {
            "timestamp": pd.to_datetime(c["startedAt"]),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c.get("baseTokenVolume", 0.0)),
        }
        for c in all_rows
    ]
    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df = df[df["timestamp"] >= pd.Timestamp(start_time)].reset_index(drop=True)

    logger.info(
        "Fetched %d unique candles spanning %s to %s.",
        len(df), df.iloc[0]["timestamp"], df.iloc[-1]["timestamp"],
    )
    return df


# --------------------------------------------------------------------------- #
# Step 2: Walk-forward signal + single-position backtest simulation
# --------------------------------------------------------------------------- #

@dataclass
class TradeResult:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str  # "STOP_LOSS" | "TAKE_PROFIT"
    return_pct: float  # net of round-trip taker fees


@dataclass
class SignalFunnel:
    candles_evaluated: int = 0
    raw_crossovers: int = 0          # single-candle crossover (pre-confirmation)
    confirmed_crossovers: int = 0    # survived the confirmation window
    suppressed_low_volatility: int = 0
    suppressed_cooldown: int = 0
    actionable_signals: int = 0
    insufficient_data_skips: int = 0


def _raw_single_candle_crossover_at(indicators: pd.DataFrame, i: int) -> Optional[str]:
    """Detect a plain (unconfirmed, confirmation_candles=1-equivalent)
    crossover using precomputed indicators, treating row `i` as the
    "current" (possibly still-forming) candle — i.e. last closed = i-1,
    previous closed = i-2. Positional (`iloc`-based) equivalent of
    `_raw_single_candle_crossover` for O(1) lookups in the backtest loop."""
    if i < 2:
        return None
    last_closed = indicators.iloc[i - 1]
    prev_closed = indicators.iloc[i - 2]
    if pd.isna(last_closed["ema_fast"]) or pd.isna(prev_closed["ema_fast"]):
        return None
    if prev_closed["ema_fast"] <= prev_closed["ema_slow"] and last_closed["ema_fast"] > last_closed["ema_slow"]:
        return "BUY"
    if prev_closed["ema_fast"] >= prev_closed["ema_slow"] and last_closed["ema_fast"] < last_closed["ema_slow"]:
        return "SELL"
    return None


def _analyze_at(
    strategy: TrendEmaStrategy,
    symbol: str,
    indicators: pd.DataFrame,
    i: int,
) -> Optional["Signal"]:
    """
    Positional equivalent of `TrendEmaStrategy.analyze()` that reuses
    precomputed indicators instead of recomputing EMA/ATR on a growing
    window — the O(n) recompute-per-step was making the walk-forward loop
    O(n^2), which is what made the grid search time out.

    Mirrors `analyze()`'s logic exactly (confirmation window, volatility
    filter, cooldown), just addressed positionally: treating row `i` as
    the boundary between "closed" history and the live/forming candle
    (i.e. last closed = i-1, previous closed = i-2), matching what
    `analyze()` would see if called on `df.iloc[:i+1]`.

    Returns None if there isn't enough history yet at this position
    (equivalent to analyze() raising InsufficientDataError).
    """
    from strategies.trend_ema import Signal

    window = strategy.confirmation_candles
    # Positional mapping from analyze()'s negative-index convention:
    # -1 -> i, -2 -> i-1, -3 -> i-2, -(1+window) -> i-window,
    # -(2+window) -> i-window-1.
    boundary_idx = i - window - 1
    if boundary_idx < 0:
        return None

    last_closed = indicators.iloc[i - 1]
    window_rows = indicators.iloc[i - window : i]
    boundary_row = indicators.iloc[boundary_idx]

    if (
        pd.isna(last_closed["ema_fast"]) or pd.isna(last_closed["ema_slow"])
        or window_rows[["ema_fast", "ema_slow"]].isna().any().any()
        or pd.isna(boundary_row["ema_fast"]) or pd.isna(boundary_row["ema_slow"])
        or pd.isna(last_closed["atr"])
    ):
        return None

    held_above = bool((window_rows["ema_fast"] > window_rows["ema_slow"]).all())
    held_below = bool((window_rows["ema_fast"] < window_rows["ema_slow"]).all())
    boundary_at_or_below = bool(boundary_row["ema_fast"] <= boundary_row["ema_slow"])
    boundary_at_or_above = bool(boundary_row["ema_fast"] >= boundary_row["ema_slow"])

    confirmed_crossed_above = held_above and boundary_at_or_below
    confirmed_crossed_below = held_below and boundary_at_or_above

    if confirmed_crossed_above:
        raw_side = "BUY"
    elif confirmed_crossed_below:
        raw_side = "SELL"
    else:
        raw_side = "HOLD"

    entry_price = float(last_closed["close"])
    atr_value = float(last_closed["atr"])
    atr_pct = (atr_value / entry_price * 100.0) if entry_price > 0 else 0.0
    last_closed_ts = pd.Timestamp(last_closed["timestamp"])
    prev_ts = pd.Timestamp(indicators.iloc[i - 2]["timestamp"])
    candle_interval = last_closed_ts - prev_ts

    side = raw_side
    reason = None

    if side != "HOLD" and atr_pct < strategy.min_atr_pct:
        reason = "low_volatility"
        side = "HOLD"

    if side != "HOLD" and strategy._cooldown_active(symbol, side, last_closed_ts, candle_interval):
        reason = "cooldown_active"
        side = "HOLD"

    sl_distance = atr_value * strategy.atr_multiplier
    tp_distance = sl_distance * strategy.risk_reward_ratio
    if side == "BUY":
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance
    elif side == "SELL":
        stop_loss = entry_price + sl_distance
        take_profit = entry_price - tp_distance
    else:
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance

    return Signal(
        symbol=symbol, side=side, entry_price=entry_price, stop_loss=stop_loss,
        take_profit=take_profit, atr_value=atr_value, timestamp=last_closed_ts.isoformat(),
        reason=reason,
    )


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    fast_ema: int,
    slow_ema: int,
    min_atr_pct: float,
    cooldown_candles: int = 8,
    confirmation_candles: int = 2,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
    risk_reward_ratio: float = 2.0,
) -> Tuple[List[TradeResult], SignalFunnel, TrendEmaStrategy]:
    """
    Walk forward through `df` one candle at a time (no lookahead — the
    signal evaluated "as of" row i only ever uses indicator values at
    rows < i, identical to what `TrendEmaStrategy.analyze()` would see if
    called on `df.iloc[:i+1]`), generating signals with the real
    TrendEmaStrategy's rules and simulating a single-position,
    long-or-short backtest against subsequent candles' high/low.

    Performance note: EMA/ATR indicators are computed ONCE up front
    (`calculate_indicators` is vectorized and purely causal — an EMA/ATR
    value at row k depends only on rows <= k, via pandas' `.ewm()`
    recursive filter), then the loop does O(1) positional lookups into
    that precomputed frame via `_analyze_at()` rather than recomputing
    indicators from scratch on a growing window at every step. This is
    what makes a multi-day, multi-combo grid search actually tractable —
    recomputing per-step would be O(n^2) and unusable at real data sizes.
    """
    strategy_logger = logging.getLogger("trend_ema_strategy")
    original_level = strategy_logger.level
    strategy_logger.setLevel(logging.WARNING)

    strategy = TrendEmaStrategy(
        fast_ema=fast_ema,
        slow_ema=slow_ema,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        risk_reward_ratio=risk_reward_ratio,
        cooldown_candles=cooldown_candles,
        min_atr_pct=min_atr_pct,
        confirmation_candles=confirmation_candles,
    )

    indicators = strategy.calculate_indicators(df)

    funnel = SignalFunnel()
    trades: List[TradeResult] = []

    position: Optional[dict] = None
    fee_frac = TAKER_FEE_PCT / 100.0
    min_start = slow_ema + atr_period + confirmation_candles + 5

    for i in range(min_start, len(df)):
        current_candle = df.iloc[i]

        if position is not None:
            side = position["side"]
            sl, tp = position["sl"], position["tp"]
            hit_sl = (current_candle["low"] <= sl) if side == "BUY" else (current_candle["high"] >= sl)
            hit_tp = (current_candle["high"] >= tp) if side == "BUY" else (current_candle["low"] <= tp)

            exit_reason = None
            exit_price = None
            if hit_sl and hit_tp:
                exit_reason, exit_price = "STOP_LOSS", sl
            elif hit_sl:
                exit_reason, exit_price = "STOP_LOSS", sl
            elif hit_tp:
                exit_reason, exit_price = "TAKE_PROFIT", tp

            if exit_reason is not None:
                if side == "BUY":
                    raw_return = (exit_price - position["entry_price"]) / position["entry_price"]
                else:
                    raw_return = (position["entry_price"] - exit_price) / position["entry_price"]
                net_return = raw_return - 2 * fee_frac

                trades.append(TradeResult(
                    side=side,
                    entry_time=position["entry_time"],
                    entry_price=position["entry_price"],
                    exit_time=current_candle["timestamp"],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    return_pct=net_return * 100.0,
                ))

                if exit_reason == "STOP_LOSS":
                    strategy.record_stop_out(symbol, side, current_candle["timestamp"])

                position = None
            else:
                continue

        raw_side = _raw_single_candle_crossover_at(indicators, i)
        if raw_side is not None:
            funnel.raw_crossovers += 1

        funnel.candles_evaluated += 1

        signal = _analyze_at(strategy, symbol, indicators, i)
        if signal is None:
            funnel.insufficient_data_skips += 1
            continue

        if signal.side == "HOLD":
            if signal.reason == "low_volatility":
                funnel.suppressed_low_volatility += 1
                funnel.confirmed_crossovers += 1
            elif signal.reason == "cooldown_active":
                funnel.suppressed_cooldown += 1
                funnel.confirmed_crossovers += 1
            continue

        funnel.confirmed_crossovers += 1
        funnel.actionable_signals += 1
        position = {
            "side": signal.side,
            "entry_time": current_candle["timestamp"],
            "entry_price": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
        }

    strategy_logger.setLevel(original_level)
    return trades, funnel, strategy


# --------------------------------------------------------------------------- #
# Step 2b: Trend-Pullback backtest engine (same performance approach —
# indicators precomputed once, O(1) positional lookups per step)
# --------------------------------------------------------------------------- #

def _analyze_pullback_at(
    strategy: TrendPullbackStrategy,
    symbol: str,
    indicators: pd.DataFrame,
    i: int,
) -> Optional["Signal"]:
    """
    Positional equivalent of `TrendPullbackStrategy.analyze()`, operating
    on precomputed indicators for O(1) lookups instead of recomputing
    EMA200/EMA20/RSI/ATR on a growing window at every step.

    Positional mapping (same convention as `_analyze_at`): treating row
    `i` as the live/forming candle, last closed = i-1, previous closed = i-2.
    """
    from strategies.trend_ema import Signal

    if i < 2:
        return None

    last_closed = indicators.iloc[i - 1]
    prev_closed = indicators.iloc[i - 2]

    required_fields = ["ema_trend", "ema_pullback", "rsi", "atr"]
    if last_closed[required_fields].isna().any() or prev_closed[required_fields].isna().any():
        return None

    close_now = float(last_closed["close"])
    ema_trend_now = float(last_closed["ema_trend"])
    ema_pullback_now = float(last_closed["ema_pullback"])
    ema_pullback_prev = float(prev_closed["ema_pullback"])
    close_prev = float(prev_closed["close"])
    rsi_prev = float(prev_closed["rsi"])

    macro_uptrend = close_now > ema_trend_now
    macro_downtrend = close_now < ema_trend_now

    pulled_back_to_ema = close_prev <= ema_pullback_prev
    pulled_back_from_ema = close_prev >= ema_pullback_prev
    closed_back_above_ema = close_now > ema_pullback_now
    closed_back_below_ema = close_now < ema_pullback_now

    long_setup = macro_uptrend and pulled_back_to_ema and closed_back_above_ema
    short_setup = macro_downtrend and pulled_back_from_ema and closed_back_below_ema

    if strategy.use_rsi_confirmation:
        long_setup = long_setup and (rsi_prev <= strategy.rsi_oversold)
        short_setup = short_setup and (rsi_prev >= strategy.rsi_overbought)

    if long_setup:
        raw_side = "BUY"
    elif short_setup:
        raw_side = "SELL"
    else:
        raw_side = "HOLD"

    entry_price = close_now
    atr_value = float(last_closed["atr"])
    atr_pct = (atr_value / entry_price * 100.0) if entry_price > 0 else 0.0
    last_closed_ts = pd.Timestamp(last_closed["timestamp"])
    prev_ts = pd.Timestamp(prev_closed["timestamp"])
    candle_interval = last_closed_ts - prev_ts

    side = raw_side
    reason = None

    if side != "HOLD" and atr_pct < strategy.min_atr_pct:
        reason = "low_volatility"
        side = "HOLD"

    if side != "HOLD" and strategy._cooldown_active(symbol, side, last_closed_ts, candle_interval):
        reason = "cooldown_active"
        side = "HOLD"

    sl_distance = atr_value * strategy.atr_multiplier
    tp_distance = sl_distance * strategy.risk_reward_ratio
    if side == "BUY":
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance
    elif side == "SELL":
        stop_loss = entry_price + sl_distance
        take_profit = entry_price - tp_distance
    else:
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance

    return Signal(
        symbol=symbol, side=side, entry_price=entry_price, stop_loss=stop_loss,
        take_profit=take_profit, atr_value=atr_value, timestamp=last_closed_ts.isoformat(),
        reason=reason,
    )


def run_backtest_pullback(
    df: pd.DataFrame,
    symbol: str,
    ema_trend: int = 200,
    ema_pullback: int = 20,
    rsi_period: int = 14,
    rsi_oversold: float = 40.0,
    rsi_overbought: float = 60.0,
    use_rsi_confirmation: bool = True,
    min_atr_pct: float = 0.12,
    cooldown_candles: int = 8,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
    risk_reward_ratio: float = 2.0,
) -> Tuple[List[TradeResult], SignalFunnel, TrendPullbackStrategy]:
    """
    Walk-forward, single-position backtest for `TrendPullbackStrategy`,
    using the exact same no-lookahead, single-position-at-a-time,
    fee-inclusive simulation mechanics as `run_backtest()` for the EMA
    crossover strategy — the only difference is which strategy's signal
    logic drives entries.
    """
    strategy_logger = logging.getLogger("trend_pullback_strategy")
    original_level = strategy_logger.level
    strategy_logger.setLevel(logging.WARNING)

    strategy = TrendPullbackStrategy(
        ema_trend=ema_trend,
        ema_pullback=ema_pullback,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        use_rsi_confirmation=use_rsi_confirmation,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        risk_reward_ratio=risk_reward_ratio,
        cooldown_candles=cooldown_candles,
        min_atr_pct=min_atr_pct,
    )

    indicators = strategy.calculate_indicators(df)

    funnel = SignalFunnel()
    trades: List[TradeResult] = []
    position: Optional[dict] = None
    fee_frac = TAKER_FEE_PCT / 100.0
    min_start = max(ema_trend, rsi_period, atr_period) + 5

    for i in range(min_start, len(df)):
        current_candle = df.iloc[i]

        if position is not None:
            side = position["side"]
            sl, tp = position["sl"], position["tp"]
            hit_sl = (current_candle["low"] <= sl) if side == "BUY" else (current_candle["high"] >= sl)
            hit_tp = (current_candle["high"] >= tp) if side == "BUY" else (current_candle["low"] <= tp)

            exit_reason = None
            exit_price = None
            if hit_sl:
                exit_reason, exit_price = "STOP_LOSS", sl
            elif hit_tp:
                exit_reason, exit_price = "TAKE_PROFIT", tp

            if exit_reason is not None:
                if side == "BUY":
                    raw_return = (exit_price - position["entry_price"]) / position["entry_price"]
                else:
                    raw_return = (position["entry_price"] - exit_price) / position["entry_price"]
                net_return = raw_return - 2 * fee_frac

                trades.append(TradeResult(
                    side=side,
                    entry_time=position["entry_time"],
                    entry_price=position["entry_price"],
                    exit_time=current_candle["timestamp"],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    return_pct=net_return * 100.0,
                ))

                if exit_reason == "STOP_LOSS":
                    strategy.record_stop_out(symbol, side, current_candle["timestamp"])

                position = None
            else:
                continue

        funnel.candles_evaluated += 1
        signal = _analyze_pullback_at(strategy, symbol, indicators, i)
        if signal is None:
            funnel.insufficient_data_skips += 1
            continue

        # For the pullback strategy, "raw" = setup detected before the
        # volatility/cooldown filters (RSI confirmation, if enabled, is
        # already baked into the setup definition itself, not a separate
        # funnel stage — see class docstring). "Confirmed" here means the
        # same thing: a real setup that reached the filter stage.
        if signal.side != "HOLD" or signal.reason is not None:
            funnel.raw_crossovers += 1
            funnel.confirmed_crossovers += 1

        if signal.side == "HOLD":
            if signal.reason == "low_volatility":
                funnel.suppressed_low_volatility += 1
            elif signal.reason == "cooldown_active":
                funnel.suppressed_cooldown += 1
            continue

        funnel.actionable_signals += 1
        position = {
            "side": signal.side,
            "entry_time": current_candle["timestamp"],
            "entry_price": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
        }

    strategy_logger.setLevel(original_level)
    return trades, funnel, strategy


# --------------------------------------------------------------------------- #
# Step 3: Performance metrics
# --------------------------------------------------------------------------- #

@dataclass
class PerformanceStats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate_pct: float = 0.0
    profit_factor: Optional[float] = None
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0


def compute_performance(trades: List[TradeResult]) -> PerformanceStats:
    stats = PerformanceStats(total_trades=len(trades))
    if not trades:
        return stats

    returns = [t.return_pct for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    stats.wins = len(wins)
    stats.losses = len(losses)
    stats.win_rate_pct = (stats.wins / stats.total_trades * 100.0) if stats.total_trades else 0.0
    stats.avg_win_pct = sum(wins) / len(wins) if wins else 0.0
    stats.avg_loss_pct = sum(losses) / len(losses) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    stats.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else None
    )

    # Equity curve assuming each trade compounds a fixed fraction of
    # capital (1x notional, no leverage) — for shape/quality assessment,
    # not a claim about dollar PnL at any particular account size.
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1.0 + r / 100.0)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak * 100.0
        max_dd = max(max_dd, drawdown)

    stats.max_drawdown_pct = max_dd
    stats.total_return_pct = (equity - 1.0) * 100.0
    return stats


# --------------------------------------------------------------------------- #
# Step 4: Grid search over EMA pairs + ATR% thresholds
# --------------------------------------------------------------------------- #

EMA_PAIR_CANDIDATES: List[Tuple[int, int]] = [(5, 13), (9, 21), (12, 26), (20, 50)]
ATR_PERCENTILES = [25, 40, 50, 60, 75]
MIN_TRADES_FOR_RECOMMENDATION = 8  # avoid recommending settings backed by 1-2 lucky trades


def measure_atr_pct_distribution(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    probe = TrendEmaStrategy(
        fast_ema=9, slow_ema=21, atr_period=atr_period, min_atr_pct=0.0, confirmation_candles=1
    )
    indicators = probe.calculate_indicators(df)
    atr_pct = (indicators["atr"] / indicators["close"] * 100.0).dropna()
    return atr_pct


def grid_search(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    results = []
    atr_pct_dist = measure_atr_pct_distribution(df)  # computed once, reused across all combos
    thresholds = sorted(set(round(float(atr_pct_dist.quantile(p / 100.0)), 4) for p in ATR_PERCENTILES))

    total_combos = len(EMA_PAIR_CANDIDATES) * len(thresholds)
    # Rough calibration from measured runs: ~1.5-2s per 1,000 candles per
    # combo on typical hardware. Purely informational — actual runtime
    # varies with data size and machine, but this keeps the user from
    # wondering if the script has hung on larger --days values.
    est_seconds = total_combos * len(df) / 1000 * 1.7
    logger.info(
        "Grid search: %d EMA pairs x %d ATR thresholds = %d combos over %d candles "
        "(~%.0fs / %.1f min estimated)...",
        len(EMA_PAIR_CANDIDATES), len(thresholds), total_combos, len(df),
        est_seconds, est_seconds / 60,
    )

    start = time.time()
    for combo_num, (fast_ema, slow_ema) in enumerate(EMA_PAIR_CANDIDATES, start=1):
        for min_atr_pct in thresholds:
            trades, funnel, _ = run_backtest(
                df, symbol, fast_ema=fast_ema, slow_ema=slow_ema, min_atr_pct=min_atr_pct,
            )
            perf = compute_performance(trades)
            results.append({
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "min_atr_pct": min_atr_pct,
                "total_trades": perf.total_trades,
                "win_rate_pct": round(perf.win_rate_pct, 2),
                "profit_factor": round(perf.profit_factor, 3) if perf.profit_factor not in (None, float("inf")) else perf.profit_factor,
                "max_drawdown_pct": round(perf.max_drawdown_pct, 3),
                "total_return_pct": round(perf.total_return_pct, 3),
            })
        elapsed = time.time() - start
        logger.info(
            "  ...EMA pair %d/%d (%d,%d) done | elapsed %.0fs",
            combo_num, len(EMA_PAIR_CANDIDATES), fast_ema, slow_ema, elapsed,
        )

    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_funnel_report(funnel: SignalFunnel, label: str = "current config: fast/slow=9/21 defaults") -> None:
    print("\n" + "=" * 70)
    print(f"SIGNAL FUNNEL ({label})")
    print("=" * 70)
    print(f"Candles evaluated (flat, indicators available): {funnel.candles_evaluated}")
    print(f"Raw setups detected:                            {funnel.raw_crossovers}")
    print(f"Setups reaching the filter stage:               {funnel.confirmed_crossovers}")
    print(f"  -> suppressed by volatility filter:            {funnel.suppressed_low_volatility}")
    print(f"  -> suppressed by cooldown:                     {funnel.suppressed_cooldown}")
    print(f"  -> actionable signals (actually traded):       {funnel.actionable_signals}")
    print(f"Skipped (insufficient data / NaN indicators):   {funnel.insufficient_data_skips}")
    print(
        "\nNote: 'raw setups' checks the entry condition at each flat bar "
        "independently; it is not necessarily a strict superset of "
        "'setups reaching the filter stage' when a strategy's own logic "
        "(e.g. a confirmation window) spans multiple bars, so don't read "
        "the ratio between them as a precise noise-reduction %%."
    )
    if funnel.confirmed_crossovers > 0:
        suppression_pct = (
            (funnel.suppressed_low_volatility + funnel.suppressed_cooldown)
            / funnel.confirmed_crossovers * 100.0
        )
        print(f"Of setups reaching the filter stage, {suppression_pct:.1f}% were further suppressed by risk controls.")


def print_performance_report(perf: PerformanceStats, label: str) -> None:
    print("\n" + "=" * 70)
    print(f"BACKTEST PERFORMANCE — {label}")
    print("=" * 70)
    print(f"Total trades:      {perf.total_trades}")
    print(f"Wins / Losses:     {perf.wins} / {perf.losses}")
    print(f"Win rate:          {perf.win_rate_pct:.2f}%")
    pf_str = f"{perf.profit_factor:.3f}" if isinstance(perf.profit_factor, float) and perf.profit_factor != float("inf") else str(perf.profit_factor)
    print(f"Profit factor:     {pf_str}")
    print(f"Avg win / loss:    {perf.avg_win_pct:+.4f}% / {perf.avg_loss_pct:+.4f}%")
    print(f"Max drawdown:      {perf.max_drawdown_pct:.3f}%")
    print(f"Total return:      {perf.total_return_pct:+.3f}% (1x notional, fees included, no leverage)")
    if perf.total_trades < MIN_TRADES_FOR_RECOMMENDATION:
        print(
            f"\n⚠️  Only {perf.total_trades} trades in this window — too few to draw strong "
            f"conclusions (recommend re-running with more days once available)."
        )


def print_recommendations(grid_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PARAMETER RECOMMENDATIONS (grid search over EMA pairs x ATR%% thresholds)")
    print("=" * 70)

    eligible = grid_df[grid_df["total_trades"] >= MIN_TRADES_FOR_RECOMMENDATION].copy()
    if eligible.empty:
        print(
            f"No parameter combination produced >= {MIN_TRADES_FOR_RECOMMENDATION} trades "
            f"over this window — the historical sample is too small/quiet to recommend "
            f"settings with confidence. Showing the full grid instead:\n"
        )
        print(grid_df.sort_values("total_return_pct", ascending=False).head(10).to_string(index=False))
        return

    eligible["pf_sort"] = eligible["profit_factor"].apply(
        lambda x: x if isinstance(x, (int, float)) and x != float("inf") else -1
    )
    ranked = eligible.sort_values(
        by=["pf_sort", "total_return_pct"], ascending=[False, False]
    )

    print(f"Top candidates (min {MIN_TRADES_FOR_RECOMMENDATION} trades required):\n")
    print(
        ranked[["fast_ema", "slow_ema", "min_atr_pct", "total_trades", "win_rate_pct",
                "profit_factor", "max_drawdown_pct", "total_return_pct"]]
        .head(10)
        .to_string(index=False)
    )

    best = ranked.iloc[0]
    print(f"\nRecommended starting point based on this window:")
    print(f"  fast_ema={int(best['fast_ema'])}, slow_ema={int(best['slow_ema'])}")
    print(f"  STRATEGY_MIN_ATR_PCT={best['min_atr_pct']}")
    print(
        f"  ({int(best['total_trades'])} trades, {best['win_rate_pct']:.1f}% win rate, "
        f"profit factor {best['profit_factor']}, max DD {best['max_drawdown_pct']:.2f}%)"
    )
    print(
        "\n⚠️  This is calibrated on ONE historical window and is a starting point, not a "
        "guarantee — re-run periodically and treat any single-window 'best' combo with "
        "healthy skepticism, especially with trade counts under a few dozen."
    )


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

async def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate trading strategies against real dYdX v4 history.")
    parser.add_argument("--symbol", default="ETH-USD")
    parser.add_argument(
        "--days", type=float, default=5.0,
        help="Days of history to fetch. Guideline: 3-7 for 1MIN/5MIN, 30-90 for 1HOUR/4HOURS "
             "(higher timeframes need much more calendar time for the same trade-count sample "
             "size, plus warmup for long EMAs like EMA200).",
    )
    parser.add_argument(
        "--resolution", default=None,
        help="Candle resolution (e.g. '5MIN', '5m', '1MIN', '15MINS', '1HOUR', '4HOURS'). "
             "Defaults to config.settings.candle_resolution.",
    )
    parser.add_argument(
        "--strategy", choices=["trend_ema", "trend_pullback", "both"], default="both",
        help="Which strategy to backtest. 'both' (default) runs both and prints a comparison table.",
    )
    parser.add_argument(
        "--ema-trend", type=int, default=None,
        help="Override trend_pullback's macro EMA period (default from config.settings.strategy_ema_trend). "
             "Useful when testing a new --resolution: e.g. EMA200 on 5M (~16.7h) vs EMA50 on 1H (~2 days) "
             "cover very different amounts of real time.",
    )
    parser.add_argument(
        "--ema-pullback", type=int, default=None,
        help="Override trend_pullback's short-term pullback EMA period (default from config.settings.strategy_ema_pullback).",
    )
    parser.add_argument("--indexer-url", default=None, help="Override the Indexer base URL (defaults to config.settings / mainnet).")
    parser.add_argument("--cache-file", default=None, help="CSV path to cache/reuse fetched candles.")
    parser.add_argument("--skip-grid-search", action="store_true", help="Skip the EMA/ATR grid search for trend_ema (faster).")
    args = parser.parse_args()

    if args.days < 1 or args.days > 90:
        parser.error(
            "--days should be between 1 and 90 (3-7 for 1MIN/5MIN; 30-90 for 1HOUR/4HOURS "
            "to get a large enough trade sample and cover EMA200's ~8+ day warmup)."
        )

    from config import settings, normalize_candle_resolution

    resolution = normalize_candle_resolution(args.resolution) if args.resolution else settings.candle_resolution

    # Resolve + strictly validate the indexer URL the same way the rest of
    # the bot does (mainnet-only unless explicitly overridden).
    if args.indexer_url:
        indexer_url = normalize_and_validate_indexer_url(args.indexer_url)
    else:
        indexer_url = settings.dydx_v4_indexer_url

    logger.info("Calibration run | symbol=%s resolution=%s days=%.1f strategy=%s",
                args.symbol, resolution, args.days, args.strategy)

    # --- Fetch (or load cached) real historical candles ---
    if args.cache_file and Path(args.cache_file).exists():
        logger.info("Loading cached candles from %s", args.cache_file)
        df = pd.read_csv(args.cache_file, parse_dates=["timestamp"])
    else:
        df = await fetch_historical_candles(args.symbol, args.days, indexer_url, resolution=resolution)
        if args.cache_file:
            df.to_csv(args.cache_file, index=False)
            logger.info("Cached candles to %s", args.cache_file)

    if len(df) < 50:
        logger.error(
            "Only %d candles available — too little data to calibrate meaningfully. "
            "Try increasing --days.", len(df),
        )
        return

    run_ema = args.strategy in ("trend_ema", "both")
    run_pullback = args.strategy in ("trend_pullback", "both")

    ema_perf = pullback_perf = None

    # --- Trend-EMA (legacy baseline) ---
    if run_ema:
        trades, funnel, _ = run_backtest(
            df, args.symbol,
            fast_ema=9, slow_ema=21,
            min_atr_pct=settings.strategy_min_atr_pct,
            cooldown_candles=settings.strategy_cooldown_candles,
            confirmation_candles=settings.strategy_confirmation_candles,
        )
        ema_perf = compute_performance(trades)
        print_funnel_report(funnel, label="TREND-EMA (baseline)")
        print_performance_report(
            ema_perf,
            label=f"trend_ema baseline, {args.symbol} @ {resolution}, {len(df)} candles (~{args.days:.1f} days)",
        )

    # --- Trend-Pullback (new strategy) ---
    if run_pullback:
        pullback_ema_trend = args.ema_trend if args.ema_trend is not None else settings.strategy_ema_trend
        pullback_ema_pullback = args.ema_pullback if args.ema_pullback is not None else settings.strategy_ema_pullback

        warmup_candles = max(pullback_ema_trend, settings.strategy_rsi_period) + 5
        if warmup_candles > len(df) * 0.5:
            logger.warning(
                "EMA(%d) warmup (~%d candles) consumes more than half of the %d candles "
                "fetched — the evaluable window will be small. Consider more --days or a "
                "smaller --ema-trend for this resolution.",
                pullback_ema_trend, warmup_candles, len(df),
            )

        trades_pb, funnel_pb, _ = run_backtest_pullback(
            df, args.symbol,
            ema_trend=pullback_ema_trend,
            ema_pullback=pullback_ema_pullback,
            rsi_period=settings.strategy_rsi_period,
            rsi_oversold=settings.strategy_rsi_oversold,
            rsi_overbought=settings.strategy_rsi_overbought,
            use_rsi_confirmation=settings.strategy_use_rsi_confirmation,
            min_atr_pct=settings.strategy_min_atr_pct,
            cooldown_candles=settings.strategy_cooldown_candles,
        )
        pullback_perf = compute_performance(trades_pb)
        print_funnel_report(funnel_pb, label="TREND-PULLBACK (new)")
        print_performance_report(
            pullback_perf,
            label=(
                f"trend_pullback (ema_trend={pullback_ema_trend}, "
                f"ema_pullback={pullback_ema_pullback}), {args.symbol} @ {resolution}, "
                f"{len(df)} candles (~{args.days:.1f} days)"
            ),
        )

    # --- Head-to-head comparison ---
    if ema_perf is not None and pullback_perf is not None:
        print("\n" + "=" * 70)
        print("HEAD-TO-HEAD COMPARISON — trend_ema (baseline) vs trend_pullback (new)")
        print("=" * 70)

        def _pf_str(pf: Optional[float]) -> str:
            if pf is None:
                return "n/a"
            if pf == float("inf"):
                return "inf"
            return f"{pf:.3f}"

        rows = [
            ("Total trades", ema_perf.total_trades, pullback_perf.total_trades),
            ("Win rate", f"{ema_perf.win_rate_pct:.2f}%", f"{pullback_perf.win_rate_pct:.2f}%"),
            ("Profit factor", _pf_str(ema_perf.profit_factor), _pf_str(pullback_perf.profit_factor)),
            ("Max drawdown", f"{ema_perf.max_drawdown_pct:.3f}%", f"{pullback_perf.max_drawdown_pct:.3f}%"),
            ("Total return", f"{ema_perf.total_return_pct:+.3f}%", f"{pullback_perf.total_return_pct:+.3f}%"),
        ]
        print(f"{'Metric':<16} {'trend_ema':>18} {'trend_pullback':>18}")
        print("-" * 54)
        for label, ema_val, pb_val in rows:
            print(f"{label:<16} {str(ema_val):>18} {str(pb_val):>18}")

        min_trades_either = min(ema_perf.total_trades, pullback_perf.total_trades)
        if min_trades_either < MIN_TRADES_FOR_RECOMMENDATION:
            print(
                f"\n⚠️  At least one strategy produced fewer than "
                f"{MIN_TRADES_FOR_RECOMMENDATION} trades over this window — treat this "
                f"comparison as directional only, not a confident verdict. Re-run with "
                f"more --days once more history is available."
            )

    if run_ema:
        atr_dist = measure_atr_pct_distribution(df)
        print("\n" + "=" * 70)
        print("REAL HISTORICAL ATR%% DISTRIBUTION")
        print("=" * 70)
        print(atr_dist.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

        # Fee-vs-move sanity check: at this resolution, is the typical
        # ATR-based take-profit distance even large enough to clear a
        # round-trip taker fee? This diagnoses fee-drag structurally,
        # rather than only discovering it after counting wins/losses.
        median_atr_pct = float(atr_dist.median())
        round_trip_fee_pct = 2 * TAKER_FEE_PCT
        typical_tp_pct = median_atr_pct * 1.5 * 2.0  # default atr_multiplier * risk_reward_ratio
        fee_burden_pct = (round_trip_fee_pct / typical_tp_pct * 100.0) if typical_tp_pct > 0 else float("inf")
        print(
            f"\nFee-drag check: median ATR%%={median_atr_pct:.4f}%%, default TP distance "
            f"(ATR x1.5 x RR2.0) ~= {typical_tp_pct:.4f}%%, round-trip taker fee = "
            f"{round_trip_fee_pct:.3f}%%."
        )
        print(f"  -> Round-trip fee consumes ~{fee_burden_pct:.1f}% of a typical winning TP move.")
        if fee_burden_pct > 15:
            print(
                "  ⚠️  Fees eating a large share of the typical target move is a structural "
                "red flag independent of entry logic — this resolution may be too short "
                "regardless of which strategy is used."
            )

    if run_ema and not args.skip_grid_search:
        logger.info(
            "Running trend_ema parameter grid search (%d EMA pairs x up to %d ATR "
            "thresholds = up to %d backtests over %d candles each)...",
            len(EMA_PAIR_CANDIDATES), len(ATR_PERCENTILES),
            len(EMA_PAIR_CANDIDATES) * len(ATR_PERCENTILES), len(df),
        )
        grid_df = grid_search(df, args.symbol)
        print_recommendations(grid_df)
    elif run_ema:
        print("\n(trend_ema grid search skipped via --skip-grid-search)")


if __name__ == "__main__":
    asyncio.run(main())