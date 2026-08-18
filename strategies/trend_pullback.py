"""
strategies/trend_pullback.py

Trend-Pullback strategy: trades pullbacks *with* the macro trend instead of
chasing raw EMA crossovers (which proved too noisy/laggy — see
scripts/calibrate_strategy.py results on 1-minute and 5-minute EMA(9/21)
crossovers).

Core logic:
    1. Trend filter (macro bias): a long-period EMA (default 200) on 5M
       candles. LONG bias only while price is above it; SHORT bias only
       while price is below it. No trades against this bias, ever.
    2. Pullback entry: instead of buying strength/selling weakness (chasing
       the breakout peak), wait for price to pull back toward a
       short-period EMA (default 20) — optionally confirmed by RSI having
       dipped into oversold/overbought territory during the pullback, as
       evidence the pullback was real rather than a one-candle wiggle.
    3. Execution trigger: enter on the first candle that closes back
       *through* the short EMA in the direction of the macro trend — i.e.
       the pullback is confirmed to be ending, not still in progress.

This keeps the same ATR-based SL/TP sizing and Signal/cooldown/volatility
risk-control interface as strategies/trend_ema.py (reuses its `Signal`
dataclass) so it's a drop-in alternative for `TradingBot` and
`scripts/calibrate_strategy.py`.

Author: Senior Python/Crypto Algorithmic Developer
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from strategies.trend_ema import (
    REQUIRED_COLUMNS,
    InsufficientDataError,
    InvalidDataFrameError,
    Signal,
    SignalSide,
)

logger = logging.getLogger("trend_pullback_strategy")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class TrendPullbackStrategy:
    """
    Macro-trend-filtered pullback entry strategy.

    LONG setup (all must hold on the last CLOSED candle, index -2, vs the
    candle before it, index -3 — index -1 / the forming candle is never
    used, to avoid repaint):
        1. close[-2] > ema_trend[-2]                (macro bias: uptrend)
        2. close[-3] <= ema_pullback[-3]             (price pulled back to/through the short EMA)
        3. close[-2] > ema_pullback[-2]              (pullback confirmed ending: closed back above)
        4. (optional) rsi[-3] <= rsi_oversold         (pullback was a real dip, not noise)

    SHORT setup is the exact mirror image (downtrend bias, pullback up
    through the short EMA, close back below it, optional RSI overbought
    confirmation).

    SL/TP sizing is ATR-based, identical mechanics to TrendEmaStrategy:
        SL distance = ATR * atr_multiplier
        TP distance = SL distance * risk_reward_ratio
    """

    def __init__(
        self,
        ema_trend: int = 200,
        ema_pullback: int = 20,
        rsi_period: int = 14,
        rsi_oversold: float = 40.0,
        rsi_overbought: float = 60.0,
        use_rsi_confirmation: bool = True,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        risk_reward_ratio: float = 2.0,
        tp_atr_multiplier: Optional[float] = None,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        use_adx_filter: bool = True,
        cooldown_candles: int = 8,
        min_atr_pct: float = 0.12,
    ) -> None:
        if ema_trend <= 0 or ema_pullback <= 0 or rsi_period <= 0 or atr_period <= 0:
            raise ValueError("ema_trend, ema_pullback, rsi_period, and atr_period must be positive")
        if ema_pullback >= ema_trend:
            raise ValueError("ema_pullback period must be strictly less than ema_trend period")
        if not (0 < rsi_oversold < 50):
            raise ValueError("rsi_oversold must be within (0, 50)")
        if not (50 < rsi_overbought < 100):
            raise ValueError("rsi_overbought must be within (50, 100)")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive")
        if risk_reward_ratio <= 0:
            raise ValueError("risk_reward_ratio must be positive")
        if tp_atr_multiplier is not None and tp_atr_multiplier <= 0:
            raise ValueError("tp_atr_multiplier must be positive when provided")
        if adx_period <= 0:
            raise ValueError("adx_period must be positive")
        if adx_threshold < 0:
            raise ValueError("adx_threshold must be non-negative")
        if cooldown_candles < 0:
            raise ValueError("cooldown_candles must be non-negative")
        if min_atr_pct < 0:
            raise ValueError("min_atr_pct must be non-negative")

        self.ema_trend = ema_trend
        self.ema_pullback = ema_pullback
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.use_rsi_confirmation = use_rsi_confirmation
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        # If set, TP is computed directly as `entry +/- tp_atr_multiplier * ATR`,
        # decoupled from the SL-distance/risk_reward_ratio framing — a
        # separate, directly tunable lever for grid search. If None
        # (default), TP falls back to the original
        # SL_distance * risk_reward_ratio behavior for backward compatibility.
        self.tp_atr_multiplier = tp_atr_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.use_adx_filter = use_adx_filter
        self.cooldown_candles = cooldown_candles
        self.min_atr_pct = min_atr_pct

        # Same bar-based (not wall-clock) cooldown mechanism as
        # TrendEmaStrategy — symbol -> side -> candle timestamp of most
        # recent stop-out.
        self._last_stop_out: Dict[str, Dict[str, pd.Timestamp]] = {}

        logger.info(
            "TrendPullbackStrategy initialized | ema_trend=%d ema_pullback=%d "
            "rsi_period=%d rsi_oversold=%.1f rsi_overbought=%.1f use_rsi_confirmation=%s "
            "atr_mult=%.2f rr=%.2f tp_atr_multiplier=%s adx_period=%d adx_threshold=%.1f "
            "use_adx_filter=%s cooldown_candles=%d min_atr_pct=%.3f%%",
            ema_trend, ema_pullback, rsi_period, rsi_oversold, rsi_overbought,
            use_rsi_confirmation, atr_multiplier, risk_reward_ratio, tp_atr_multiplier,
            adx_period, adx_threshold, use_adx_filter, cooldown_candles, min_atr_pct,
        )

    # ------------------------------------------------------------------ #
    # Cooldown state management (identical semantics to TrendEmaStrategy)
    # ------------------------------------------------------------------ #

    def record_stop_out(self, symbol: str, side: str, timestamp: pd.Timestamp) -> None:
        """Arm a post-stop-out cooldown for `symbol`+`side`, suppressing
        new same-direction entries for `cooldown_candles` candles."""
        normalized_side = self._normalize_side(side)
        self._last_stop_out.setdefault(symbol, {})[normalized_side] = pd.Timestamp(timestamp)
        logger.info(
            "COOLDOWN ARMED | %s %s side suppressed for %d candles starting %s",
            symbol, normalized_side, self.cooldown_candles, timestamp,
        )

    def reset_cooldown(self, symbol: str, side: Optional[str] = None) -> None:
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
        upper = side.upper()
        if upper in ("BUY", "LONG"):
            return "LONG"
        if upper in ("SELL", "SHORT"):
            return "SHORT"
        raise ValueError(f"invalid side for cooldown tracking: {side}")

    def _cooldown_active(
        self, symbol: str, signal_side: SignalSide, last_closed_ts: pd.Timestamp,
        interval: Optional[pd.Timedelta],
    ) -> bool:
        if self.cooldown_candles <= 0 or signal_side == "HOLD":
            return False
        side_key = "LONG" if signal_side == "BUY" else "SHORT"
        last_stop_out_ts = self._last_stop_out.get(symbol, {}).get(side_key)
        if last_stop_out_ts is None:
            return False

        elapsed = pd.Timestamp(last_closed_ts) - last_stop_out_ts
        if elapsed.total_seconds() < 0:
            return True

        if interval is None or interval.total_seconds() <= 0:
            interval = pd.Timedelta(minutes=5)

        elapsed_candles = elapsed / interval
        return elapsed_candles < self.cooldown_candles

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise InvalidDataFrameError(f"DataFrame is missing required column(s): {missing}")

    def _min_required_rows(self) -> int:
        # ADX needs roughly 2x its period for a stable (Wilder-double-
        # smoothed) value; EMA(200) is usually the larger constraint, but
        # take whichever warmup is longest. +5 buffer for the boundary/
        # comparison candles used in the entry logic.
        return max(self.ema_trend, self.rsi_period, self.atr_period, self.adx_period * 2) + 5

    # ------------------------------------------------------------------ #
    # Indicator calculation
    # ------------------------------------------------------------------ #

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
        range1 = high - low
        range2 = (high - prev_close).abs()
        range3 = (low - prev_close).abs()
        true_range = pd.concat([range1, range2, range3], axis=1).max(axis=1)
        return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    def _adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Standard Wilder ADX (Average Directional Index): measures trend
        STRENGTH (not direction) on a 0-100 scale. Used here purely as a
        gate — high ADX means the market is trending strongly enough for
        a trend-following pullback entry to be meaningful; low ADX means
        consolidation/chop, where pullback entries tend to whipsaw.

        Standard construction:
            +DM = max(high[t]-high[t-1], 0) if it exceeds -DM, else 0
            -DM = max(low[t-1]-low[t], 0) if it exceeds +DM, else 0
            TR  = true range (same as _atr's true_range, unsmoothed)
            +DI = 100 * Wilder_smooth(+DM) / Wilder_smooth(TR)
            -DI = 100 * Wilder_smooth(-DM) / Wilder_smooth(TR)
            DX  = 100 * |+DI - -DI| / (+DI + -DI)
            ADX = Wilder_smooth(DX)
        """
        import numpy as np

        high, low, close = df["high"], df["low"], df["close"]
        prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
        )

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        smoothed_tr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        smoothed_plus_dm = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = 100.0 * (smoothed_plus_dm / smoothed_tr)
            minus_di = 100.0 * (smoothed_minus_dm / smoothed_tr)
            di_sum = plus_di + minus_di
            dx = 100.0 * (plus_di - minus_di).abs() / di_sum

        dx = dx.where(di_sum != 0, other=0.0)
        adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        # Preserve NaN through the full double-warmup period.
        adx = adx.where(smoothed_tr.notna() & dx.notna(), other=pd.NA).astype("float64")
        return adx

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Standard Wilder's RSI: RS = avg_gain / avg_loss (Wilder-smoothed
        via ewm(alpha=1/period)), RSI = 100 - 100 / (1 + RS)."""
        import numpy as np

        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        # Avoid division by zero: where avg_loss is 0, RSI is 100 (pure
        # gains) unless avg_gain is also 0 (flat), in which case RSI is 50
        # (neutral).
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(avg_loss != 0, other=100.0)
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), other=50.0)
        # Preserve NaN during the indicator warm-up period (avg_gain/loss
        # are NaN until `min_periods` is reached).
        rsi = rsi.where(avg_gain.notna() & avg_loss.notna(), other=pd.NA).astype("float64")
        return rsi

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute `ema_trend`, `ema_pullback`, `rsi`, `atr`, and `adx` columns
        on a copy of `df`. Expects columns:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        """
        self._validate_columns(df)
        out = df.copy().reset_index(drop=True)
        out["ema_trend"] = self._ema(out["close"], self.ema_trend)
        out["ema_pullback"] = self._ema(out["close"], self.ema_pullback)
        out["rsi"] = self._rsi(out["close"], self.rsi_period)
        out["atr"] = self._atr(out, self.atr_period)
        out["adx"] = self._adx(out, self.adx_period)
        return out

    # ------------------------------------------------------------------ #
    # Signal analysis
    # ------------------------------------------------------------------ #

    def analyze(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Analyze OHLCV data and produce a trading Signal using the
        trend-pullback rules described in the class docstring.

        Raises:
            InvalidDataFrameError: if required columns are missing.
            InsufficientDataError: if there isn't enough history to warm
                up the EMA(ema_trend)/RSI/ATR indicators reliably.
        """
        self._validate_columns(df)

        min_rows = self._min_required_rows()
        if len(df) < min_rows:
            raise InsufficientDataError(
                f"need at least {min_rows} rows to analyze "
                f"(max(ema_trend={self.ema_trend}, rsi_period={self.rsi_period}, "
                f"atr_period={self.atr_period}, adx_period*2={self.adx_period * 2}) + 5), got {len(df)}"
            )

        indicators = self.calculate_indicators(df)

        last_closed = indicators.iloc[-2]
        prev_closed = indicators.iloc[-3]

        required_fields = ["ema_trend", "ema_pullback", "rsi", "atr", "adx"]
        if last_closed[required_fields].isna().any() or prev_closed[required_fields].isna().any():
            raise InsufficientDataError(
                "indicators contain NaN values on the evaluated candles — "
                "provide more historical rows before the analysis window"
            )

        close_now = float(last_closed["close"])
        ema_trend_now = float(last_closed["ema_trend"])
        ema_pullback_now = float(last_closed["ema_pullback"])
        ema_pullback_prev = float(prev_closed["ema_pullback"])
        close_prev = float(prev_closed["close"])
        rsi_prev = float(prev_closed["rsi"])
        adx_now = float(last_closed["adx"])

        macro_uptrend = close_now > ema_trend_now
        macro_downtrend = close_now < ema_trend_now

        pulled_back_to_ema = close_prev <= ema_pullback_prev
        pulled_back_from_ema = close_prev >= ema_pullback_prev
        closed_back_above_ema = close_now > ema_pullback_now
        closed_back_below_ema = close_now < ema_pullback_now

        long_setup = macro_uptrend and pulled_back_to_ema and closed_back_above_ema
        short_setup = macro_downtrend and pulled_back_from_ema and closed_back_below_ema

        if self.use_rsi_confirmation:
            long_setup = long_setup and (rsi_prev <= self.rsi_oversold)
            short_setup = short_setup and (rsi_prev >= self.rsi_overbought)

        raw_side: SignalSide
        if long_setup:
            raw_side = "BUY"
        elif short_setup:
            raw_side = "SELL"
        else:
            raw_side = "HOLD"

        entry_price = close_now
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
        try:
            candle_interval: Optional[pd.Timedelta] = last_closed_ts - pd.Timestamp(prev_timestamp)
        except Exception:  # noqa: BLE001
            candle_interval = None

        side: SignalSide = raw_side
        reason: Optional[str] = None

        # ADX gate: only allow entries while the market is trending
        # strongly enough (ADX above threshold) for a trend-following
        # pullback to be meaningful. Applied BEFORE the volatility filter
        # since it's a distinct concept (trend strength vs. raw price
        # movement magnitude) — a market can have high ATR% while still
        # chopping directionlessly (low ADX), or low ATR% within a
        # genuine slow grind (higher ADX).
        if side != "HOLD" and self.use_adx_filter and adx_now < self.adx_threshold:
            logger.info(
                "SIGNAL FILTERED (weak trend / ADX) | %s %s candidate suppressed: "
                "adx=%.2f < adx_threshold=%.2f",
                symbol, side, adx_now, self.adx_threshold,
            )
            reason = "weak_trend_adx"
            side = "HOLD"

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
        # ATR-based TP: if tp_atr_multiplier is set, TP is a direct,
        # independently tunable N*ATR distance from entry rather than
        # being derived from the SL distance via risk_reward_ratio.
        tp_distance = (
            atr_value * self.tp_atr_multiplier
            if self.tp_atr_multiplier is not None
            else sl_distance * self.risk_reward_ratio
        )

        if side == "BUY":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        elif side == "SELL":
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        else:
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
            "SIGNAL GENERATED | %s %s entry=%.4f SL=%.4f TP=%.4f ATR=%.4f ADX=%.2f "
            "(atr_pct=%.4f%% ema_trend=%.4f ema_pullback=%.4f rsi_prev=%.2f "
            "macro=%s raw_side=%s reason=%s)",
            symbol, side, entry_price, stop_loss, take_profit, atr_value, adx_now, atr_pct,
            ema_trend_now, ema_pullback_now, rsi_prev,
            "UP" if macro_uptrend else ("DOWN" if macro_downtrend else "FLAT"),
            raw_side, reason,
        )

        return signal


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

def _build_synthetic_pullback_ohlcv() -> pd.DataFrame:
    """
    Synthetic 5-min series: a sustained uptrend (price above EMA200)
    with a clean short pullback toward EMA20 that then resumes upward —
    the canonical trend-pullback long setup.
    """
    import numpy as np

    rng = pd.date_range(start="2026-01-01", periods=260, freq="5min", tz="UTC")
    price = 2500.0
    prices = []

    # Long, gentle uptrend to get EMA200 solidly below price (220 candles).
    for _ in range(220):
        price += 1.2
        prices.append(price)

    # Sharp short pullback toward/through EMA20 (8 candles down).
    for _ in range(8):
        price -= 3.0
        prices.append(price)

    # Resume uptrend, closing back above EMA20 (32 candles).
    for _ in range(32):
        price += 2.0
        prices.append(price)

    rows = []
    prev_close = prices[0] - 1.2
    for ts, close in zip(rng, prices):
        open_ = prev_close
        high = max(open_, close) + 1.0
        low = min(open_, close) - 1.0
        rows.append({"timestamp": ts, "open": open_, "high": high, "low": low,
                     "close": close, "volume": 500.0})
        prev_close = close

    return pd.DataFrame(rows)


def _demo() -> None:
    strategy = TrendPullbackStrategy(
        ema_trend=200, ema_pullback=20, rsi_period=14,
        rsi_oversold=40, rsi_overbought=60,
        use_rsi_confirmation=False,  # kept off here purely for a clean, deterministic demo
    )
    full_df = _build_synthetic_pullback_ohlcv()

    # Scan for the exact candle where the pullback-entry condition first
    # fires, then truncate the series so that candle lands at index -2 —
    # deterministic instead of guessing synthetic-series lengths.
    indicators = strategy.calculate_indicators(full_df)
    entry_idx = None
    for i in range(strategy._min_required_rows(), len(indicators)):
        row, prev = indicators.iloc[i], indicators.iloc[i - 1]
        if row[["ema_trend", "ema_pullback", "rsi"]].isna().any():
            continue
        macro_up = row["close"] > row["ema_trend"]
        pulled_back = prev["close"] <= prev["ema_pullback"]
        confirmed = row["close"] > row["ema_pullback"]
        rsi_ok = (prev["rsi"] <= strategy.rsi_oversold) if strategy.use_rsi_confirmation else True
        if macro_up and pulled_back and confirmed and rsi_ok:
            entry_idx = i
            break

    if entry_idx is None:
        print("No qualifying pullback entry found in the synthetic series.")
        df = full_df
    else:
        df = full_df.iloc[: entry_idx + 2].reset_index(drop=True)  # entry candle at index -2

    print(f"Synthetic OHLCV rows: {len(df)}")

    indicators_df = strategy.calculate_indicators(df)
    print("\n=== Last 10 rows with indicators ===")
    print(
        indicators_df[["timestamp", "close", "ema_trend", "ema_pullback", "rsi", "atr"]]
        .tail(10)
        .to_string(index=False)
    )

    signal = strategy.analyze(symbol="ETH-USD", df=df)
    print("\n=== Generated Signal ===")
    print(signal)
    print(signal.to_dict())


if __name__ == "__main__":
    _demo()