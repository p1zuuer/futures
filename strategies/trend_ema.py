"""
strategies/trend_ema.py

EMA Crossover trend strategy with dynamic ATR-based Stop-Loss / Take-Profit
sizing. Pure pandas + manual math (no TA-Lib / pandas-ta dependency) so it
runs in any environment.

Signal Rules:
    BUY  -> fast EMA crosses ABOVE slow EMA on the last CLOSED candle.
    SELL -> fast EMA crosses BELOW slow EMA on the last CLOSED candle.
    HOLD -> no crossover on the last closed candle.

Crossover is evaluated strictly on index -2 vs index -3 (the last fully
closed candle relative to the one before it), never on index -1 (the
current/open candle), to avoid repainting signals as the live candle forms.

Author: Senior Python/Crypto Algorithmic Developer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

import pandas as pd

logger = logging.getLogger("trend_ema_strategy")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


SignalSide = Literal["BUY", "SELL", "HOLD"]

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class StrategyError(Exception):
    """Base exception for strategy-level errors."""


class InsufficientDataError(StrategyError):
    """Raised when the input DataFrame does not have enough rows to analyze."""


class InvalidDataFrameError(StrategyError):
    """Raised when the input DataFrame is missing required OHLCV columns."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Signal:
    """Structured trading signal with risk management levels."""

    symbol: str
    side: SignalSide
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    timestamp: str
    reason: Optional[str] = None
    """Populated when side == "HOLD" and the crossover was actively
    suppressed by a risk filter (e.g. "cooldown_active",
    "low_volatility", "unconfirmed_crossover"), vs. simply "no_crossover"
    when there was nothing to filter in the first place. Useful for
    logging/telemetry without changing the Signal's shape."""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 6),
            "stop_loss": round(self.stop_loss, 6),
            "take_profit": round(self.take_profit, 6),
            "atr_value": round(self.atr_value, 6),
            "timestamp": self.timestamp,
            "reason": self.reason,
        }

    def __str__(self) -> str:
        suffix = f" ({self.reason})" if self.reason else ""
        return (
            f"Signal({self.symbol} | {self.side}{suffix} | entry={self.entry_price:.4f} "
            f"SL={self.stop_loss:.4f} TP={self.take_profit:.4f} "
            f"ATR={self.atr_value:.4f} @ {self.timestamp})"
        )


class TrendEmaStrategy:
    """
    EMA(fast) / EMA(slow) crossover strategy with ATR-based dynamic
    Stop-Loss / Take-Profit distances, plus risk controls to reduce
    whipsaw losses in choppy/low-volatility conditions:

        1. Post-stop-out cooldown: after a position on this symbol+side
           is stopped out, crossovers in the same direction are ignored
           for `cooldown_candles` candles.
        2. Volatility filter: crossovers are ignored when ATR relative to
           price is below `min_atr_pct` — i.e. the market is too dead/flat
           for a trend-following signal to be meaningful.
        3. Crossover confirmation: instead of firing on the very first
           candle a crossover appears, the crossover must persist for
           `confirmation_candles` consecutive closed candles before a
           signal fires. This filters out single-candle noise crosses
           that immediately revert (classic whipsaw pattern).

    SL distance = ATR * atr_multiplier
    TP distance = SL distance * risk_reward_ratio

    For BUY:  stop_loss = entry - SL_distance, take_profit = entry + TP_distance
    For SELL: stop_loss = entry + SL_distance, take_profit = entry - TP_distance
    """

    def __init__(
        self,
        fast_ema: int = 9,
        slow_ema: int = 21,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        risk_reward_ratio: float = 2.0,
        cooldown_candles: int = 8,
        min_atr_pct: float = 0.12,
        confirmation_candles: int = 2,
    ) -> None:
        if fast_ema <= 0 or slow_ema <= 0 or atr_period <= 0:
            raise ValueError("fast_ema, slow_ema, and atr_period must be positive integers")
        if fast_ema >= slow_ema:
            raise ValueError("fast_ema period must be strictly less than slow_ema period")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive")
        if risk_reward_ratio <= 0:
            raise ValueError("risk_reward_ratio must be positive")
        if cooldown_candles < 0:
            raise ValueError("cooldown_candles must be non-negative")
        if min_atr_pct < 0:
            raise ValueError("min_atr_pct must be non-negative")
        if confirmation_candles < 1:
            raise ValueError("confirmation_candles must be at least 1")

        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.cooldown_candles = cooldown_candles
        self.min_atr_pct = min_atr_pct
        self.confirmation_candles = confirmation_candles

        # symbol -> side -> candle timestamp of the most recent stop-out.
        # Bar-based (not wall-clock) cooldown: elapsed candles is computed
        # from the inferred candle interval, so behavior is consistent
        # regardless of how often analyze() is polled.
        self._last_stop_out: Dict[str, Dict[str, pd.Timestamp]] = {}

        logger.info(
            "TrendEmaStrategy initialized | fast=%d slow=%d atr_period=%d "
            "atr_mult=%.2f rr=%.2f cooldown_candles=%d min_atr_pct=%.3f%% "
            "confirmation_candles=%d",
            fast_ema, slow_ema, atr_period, atr_multiplier, risk_reward_ratio,
            cooldown_candles, min_atr_pct, confirmation_candles,
        )

    # ------------------------------------------------------------------ #
    # Cooldown state management
    # ------------------------------------------------------------------ #

    def record_stop_out(self, symbol: str, side: str, timestamp: pd.Timestamp) -> None:
        """
        Notify the strategy that a stop-loss was hit for `symbol` on the
        given `side` ("LONG"/"SHORT" or "BUY"/"SELL" — normalized
        internally) at `timestamp`. Suppresses new signals in the same
        direction for `cooldown_candles` candles from this point.

        The caller (TradingBot) should call this from wherever it detects
        a STOP_LOSS-triggered position close.
        """
        normalized_side = self._normalize_side(side)
        self._last_stop_out.setdefault(symbol, {})[normalized_side] = pd.Timestamp(timestamp)
        logger.info(
            "COOLDOWN ARMED | %s %s side suppressed for %d candles starting %s",
            symbol, normalized_side, self.cooldown_candles, timestamp,
        )

    def reset_cooldown(self, symbol: str, side: Optional[str] = None) -> None:
        """Clear cooldown state for `symbol` (and optionally a specific
        side only). Mainly useful for tests or manual overrides."""
        if symbol not in self._last_stop_out:
            return
        if side is None:
            self._last_stop_out.pop(symbol, None)
        else:
            self._last_stop_out[symbol].pop(self._normalize_side(side), None)

    def get_cooldown_state(self) -> dict:
        """
        Serialize all active cooldowns to a JSON-safe dict:
        {symbol: {side: iso_timestamp}}. Used for restart-safety
        persistence — cooldown state lives only in this in-process dict
        otherwise, and would be silently wiped by a process restart.
        """
        return {
            symbol: {side: ts.isoformat() for side, ts in sides.items()}
            for symbol, sides in self._last_stop_out.items()
        }

    def restore_cooldown_state(self, state: dict) -> None:
        """Restore cooldowns previously produced by `get_cooldown_state()`."""
        restored: Dict[str, Dict[str, pd.Timestamp]] = {}
        try:
            for symbol, sides in state.items():
                restored[symbol] = {
                    side: pd.Timestamp(ts) for side, ts in sides.items()
                }
            self._last_stop_out = restored
            total = sum(len(sides) for sides in restored.values())
            logger.info("Cooldown state restored | %d active cooldown(s) across %d symbol(s)",
                        total, len(restored))
        except (AttributeError, ValueError, TypeError) as exc:
            logger.error(
                "Failed to restore cooldown state (%s) — starting with no active "
                "cooldowns instead of a corrupted/partial one.", exc,
            )

    @staticmethod
    def _normalize_side(side: str) -> str:
        """Map BUY/LONG -> LONG and SELL/SHORT -> SHORT for consistent
        cooldown bucketing regardless of which vocabulary the caller uses."""
        upper = side.upper()
        if upper in ("BUY", "LONG"):
            return "LONG"
        if upper in ("SELL", "SHORT"):
            return "SHORT"
        raise ValueError(f"invalid side for cooldown tracking: {side}")

    def _cooldown_active(
        self, symbol: str, signal_side: SignalSide, last_closed_ts: pd.Timestamp, interval: Optional[pd.Timedelta]
    ) -> bool:
        if self.cooldown_candles <= 0 or signal_side == "HOLD":
            return False
        side_key = "LONG" if signal_side == "BUY" else "SHORT"
        last_stop_out_ts = self._last_stop_out.get(symbol, {}).get(side_key)
        if last_stop_out_ts is None:
            return False

        elapsed = pd.Timestamp(last_closed_ts) - last_stop_out_ts
        if elapsed.total_seconds() < 0:
            # Stop-out recorded after this candle (stale/out-of-order call) —
            # treat as still within cooldown to be safe.
            return True

        if interval is None or interval.total_seconds() <= 0:
            # Can't infer candle spacing (e.g. exactly 2 rows) — fall back
            # to a conservative default of 1 minute per candle.
            interval = pd.Timedelta(minutes=1)

        elapsed_candles = elapsed / interval
        return elapsed_candles < self.cooldown_candles

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise InvalidDataFrameError(
                f"DataFrame is missing required column(s): {missing}"
            )

    def _min_required_rows(self) -> int:
        # +5 base buffer, plus enough extra rows to evaluate
        # `confirmation_candles` consecutive closed candles before the
        # live one (indices -2 .. -(2+confirmation_candles)).
        return self.slow_ema + 5 + self.confirmation_candles

    # ------------------------------------------------------------------ #
    # Indicator calculation
    # ------------------------------------------------------------------ #

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        """Standard exponential moving average (adjust=False mirrors typical
        charting-platform EMA recursion: EMA_t = price*k + EMA_{t-1}*(1-k))."""
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Average True Range via Wilder's smoothing (RMA), computed manually:

            TR = max(high - low, |high - prev_close|, |low - prev_close|)
            ATR = Wilder-smoothed moving average of TR over `period`
        """
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        range1 = high - low
        range2 = (high - prev_close).abs()
        range3 = (low - prev_close).abs()

        true_range = pd.concat([range1, range2, range3], axis=1).max(axis=1)

        # Wilder's smoothing is equivalent to an EMA with alpha = 1/period.
        atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        return atr

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute `ema_fast`, `ema_slow`, and `atr` columns on a copy of `df`.

        Expects columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        Returns a new DataFrame (input is not mutated in place).
        """
        self._validate_columns(df)

        out = df.copy().reset_index(drop=True)
        out["ema_fast"] = self._ema(out["close"], self.fast_ema)
        out["ema_slow"] = self._ema(out["close"], self.slow_ema)
        out["atr"] = self._atr(out, self.atr_period)

        return out

    # ------------------------------------------------------------------ #
    # Signal analysis
    # ------------------------------------------------------------------ #

    def analyze(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Analyze OHLCV data and produce a trading Signal.

        Pipeline (each stage can suppress the signal down to HOLD):
          1. Confirmed crossover detection — the fast/slow EMA relationship
             must have just flipped and held for `confirmation_candles`
             consecutive CLOSED candles (indices -2 .. -(1+confirmation_candles)),
             with the candle immediately before that window showing the
             opposite relationship. With confirmation_candles=1 this is
             identical to a plain single-candle crossover check. Index -1
             (the forming/open candle) is never used, to avoid repaint.
          2. Volatility filter — if ATR relative to price is below
             `min_atr_pct`, any crossover is suppressed (dead/flat chop).
          3. Post-stop-out cooldown — if a stop-loss was recently recorded
             for this symbol+side via `record_stop_out()`, crossovers in
             that same direction are suppressed until `cooldown_candles`
             candles have elapsed.

        Raises:
            InvalidDataFrameError: if required columns are missing.
            InsufficientDataError: if there are not enough rows to compute
                indicators reliably.
        """
        self._validate_columns(df)

        min_rows = self._min_required_rows()
        if len(df) < min_rows:
            raise InsufficientDataError(
                f"need at least {min_rows} rows to analyze "
                f"(slow_ema={self.slow_ema} + 5 + confirmation_candles="
                f"{self.confirmation_candles}), got {len(df)}"
            )

        indicators = self.calculate_indicators(df)

        # Last closed candle = index -2 ; candle before it = index -3.
        last_closed = indicators.iloc[-2]
        prev_closed = indicators.iloc[-3]

        window = self.confirmation_candles
        # Confirmation window: indices -2 .. -(1+window); boundary candle
        # right before the window is -(2+window).
        window_rows = indicators.iloc[-(1 + window) : -1]
        boundary_row = indicators.iloc[-(2 + window)]

        nan_check_rows = pd.concat([window_rows, boundary_row.to_frame().T])
        if (
            nan_check_rows[["ema_fast", "ema_slow"]].isna().any().any()
            or pd.isna(last_closed["atr"])
        ):
            raise InsufficientDataError(
                "indicators contain NaN values on the evaluated candles — "
                "provide more historical rows before the analysis window"
            )

        fast_now, slow_now = last_closed["ema_fast"], last_closed["ema_slow"]
        fast_prev, slow_prev = prev_closed["ema_fast"], prev_closed["ema_slow"]

        held_above = bool((window_rows["ema_fast"] > window_rows["ema_slow"]).all())
        held_below = bool((window_rows["ema_fast"] < window_rows["ema_slow"]).all())
        boundary_at_or_below = bool(boundary_row["ema_fast"] <= boundary_row["ema_slow"])
        boundary_at_or_above = bool(boundary_row["ema_fast"] >= boundary_row["ema_slow"])

        confirmed_crossed_above = held_above and boundary_at_or_below
        confirmed_crossed_below = held_below and boundary_at_or_above

        raw_side: SignalSide
        if confirmed_crossed_above:
            raw_side = "BUY"
        elif confirmed_crossed_below:
            raw_side = "SELL"
        else:
            raw_side = "HOLD"

        entry_price = float(last_closed["close"])
        atr_value = float(last_closed["atr"])
        atr_pct = (atr_value / entry_price * 100.0) if entry_price > 0 else 0.0

        candle_timestamp = last_closed["timestamp"]
        prev_timestamp = prev_closed["timestamp"]
        if isinstance(candle_timestamp, (pd.Timestamp, datetime)):
            last_closed_ts = pd.Timestamp(candle_timestamp)
            timestamp_str = last_closed_ts.isoformat()
        else:
            last_closed_ts = pd.Timestamp(str(candle_timestamp))
            timestamp_str = str(candle_timestamp)
        candle_interval: Optional[pd.Timedelta]
        try:
            candle_interval = last_closed_ts - pd.Timestamp(prev_timestamp)
        except Exception:  # noqa: BLE001
            candle_interval = None

        # --- Filter pipeline (only meaningful if a crossover was found) ---
        side: SignalSide = raw_side
        reason: Optional[str] = None

        if side != "HOLD" and atr_pct < self.min_atr_pct:
            logger.info(
                "SIGNAL FILTERED (low volatility) | %s %s candidate suppressed: "
                "atr_pct=%.4f%% < min_atr_pct=%.4f%%",
                symbol, side, atr_pct, self.min_atr_pct,
            )
            reason = "low_volatility"
            side = "HOLD"

        if side != "HOLD" and self._cooldown_active(symbol, side, last_closed_ts, candle_interval):
            side_key = "LONG" if side == "BUY" else "SHORT"
            last_stop_out_ts = self._last_stop_out[symbol][side_key]
            logger.info(
                "SIGNAL FILTERED (cooldown active) | %s %s candidate suppressed: "
                "last stop-out at %s, cooldown=%d candles",
                symbol, side, last_stop_out_ts, self.cooldown_candles,
            )
            reason = "cooldown_active"
            side = "HOLD"

        sl_distance = atr_value * self.atr_multiplier
        tp_distance = sl_distance * self.risk_reward_ratio

        if side == "BUY":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        elif side == "SELL":
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        else:
            # HOLD: still surface informative SL/TP anchored to a long bias
            # (entry_price is not an actionable fill in this case).
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance

        signal = Signal(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=atr_value,
            timestamp=timestamp_str,
            reason=reason,
        )

        logger.info(
            "SIGNAL GENERATED | %s %s entry=%.4f SL=%.4f TP=%.4f ATR=%.4f "
            "(atr_pct=%.4f%% fast=%.4f slow=%.4f prev_fast=%.4f prev_slow=%.4f "
            "raw_side=%s reason=%s)",
            symbol, side, entry_price, stop_loss, take_profit, atr_value,
            atr_pct, fast_now, slow_now, fast_prev, slow_prev, raw_side, reason,
        )

        return signal


# --------------------------------------------------------------------------- #
# Demo: synthetic trend-reversal OHLCV series
# --------------------------------------------------------------------------- #

def _build_synthetic_ohlcv() -> pd.DataFrame:
    """
    Build a synthetic OHLCV series that downtrends and then reverses into an
    uptrend, guaranteeing a fast/slow EMA crossover appears mid-series.
    """
    import numpy as np

    rng = pd.date_range(start="2026-01-01", periods=37, freq="1h", tz="UTC")

    prices = []
    price = 3100.0

    # Phase 1: downtrend (30 candles) — fast EMA drifts below slow EMA.
    for _ in range(30):
        price -= 6.0
        prices.append(price)

    # Phase 2: sharp reversal (7 candles) — fast EMA crosses back above slow
    # EMA and HOLDS above it for at least confirmation_candles closed
    # candles before the tail (live) candle, so the confirmed-crossover
    # signal lands on the last-closed candle (index -2) evaluated by
    # analyze() with the default confirmation_candles=2.
    for _ in range(7):
        price += 32.0
        prices.append(price)

    rows = []
    prev_close = prices[0] + 6.0
    for ts, close in zip(rng, prices):
        open_ = prev_close
        high = max(open_, close) + 3.0
        low = min(open_, close) - 3.0
        volume = 1000.0
        rows.append(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        prev_close = close

    return pd.DataFrame(rows)


def _demo() -> None:
    strategy = TrendEmaStrategy(
        fast_ema=9,
        slow_ema=21,
        atr_period=14,
        atr_multiplier=1.5,
        risk_reward_ratio=2.0,
    )

    df = _build_synthetic_ohlcv()
    print(f"Synthetic OHLCV rows: {len(df)}")
    print(df.tail(8).to_string(index=False))

    indicators_df = strategy.calculate_indicators(df)
    print("\n=== Last 8 rows with indicators ===")
    print(
        indicators_df[["timestamp", "close", "ema_fast", "ema_slow", "atr"]]
        .tail(8)
        .to_string(index=False)
    )

    signal = strategy.analyze(symbol="ETH-USD", df=df)

    print("\n=== Generated Signal ===")
    print(signal)
    print(signal.to_dict())


if __name__ == "__main__":
    _demo()