"""
strategies/trend_pullback.py

Trend-Pullback strategy: trades pullbacks *with* the macro trend instead of
chasing raw EMA crossovers.

Optimized core strategy logic:
    1. Trend filter (macro bias): long-period EMA (default 200) on candles.
    2. Dynamic Volatility / ATR Stop-Loss & Take-Profit: Adjust SL/TP dynamically
       to market structure using ATR multipliers.
    3. RSI / Stochastic Filter: RSI pullback indicator (buying oversold / selling overbought).
    4. Volume Confirmation: Volume spike check (volume > moving average of volume * threshold)
       to avoid fake pullbacks during low volume.
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
    Macro-trend-filtered pullback entry strategy with dynamic ATR stops,
    RSI pullback confirmation, and volume spike confirmation.
    """

    def __init__(
        self,
        ema_trend: int = 200,
        ema_pullback: int = 20,
        rsi_period: int = 14,
        rsi_oversold: float = 45.0,
        rsi_overbought: float = 55.0,
        use_rsi_confirmation: bool = True,
        atr_period: int = 14,
        atr_multiplier_sl: float = 1.5,
        atr_multiplier_tp: float = 2.5,
        use_dynamic_atr_stops: bool = True,
        risk_reward_ratio: float = 2.0,
        tp_atr_multiplier: Optional[float] = None,
        adx_period: int = 14,
        adx_threshold: float = 18.0,
        use_adx_filter: bool = True,
        volume_ma_period: int = 20,
        volume_spike_threshold: float = 1.1,
        use_volume_confirmation: bool = True,
        cooldown_candles: int = 5,
        min_atr_pct: float = 0.08,
    ) -> None:
        if ema_trend <= 0 or ema_pullback <= 0 or rsi_period <= 0 or atr_period <= 0:
            raise ValueError("ema_trend, ema_pullback, rsi_period, and atr_period must be positive")
        if ema_pullback >= ema_trend:
            raise ValueError("ema_pullback period must be strictly less than ema_trend period")
        if not (0 < rsi_oversold < 50):
            raise ValueError("rsi_oversold must be within (0, 50)")
        if not (50 < rsi_overbought < 100):
            raise ValueError("rsi_overbought must be within (50, 100)")
        if atr_multiplier_sl <= 0 or atr_multiplier_tp <= 0:
            raise ValueError("atr multipliers must be positive")

        self.ema_trend = ema_trend
        self.ema_pullback = ema_pullback
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.use_rsi_confirmation = use_rsi_confirmation
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.use_dynamic_atr_stops = use_dynamic_atr_stops
        self.risk_reward_ratio = risk_reward_ratio
        self.tp_atr_multiplier = tp_atr_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.use_adx_filter = use_adx_filter
        self.volume_ma_period = volume_ma_period
        self.volume_spike_threshold = volume_spike_threshold
        self.use_volume_confirmation = use_volume_confirmation
        self.cooldown_candles = cooldown_candles
        self.min_atr_pct = min_atr_pct

        self._last_stop_out: Dict[str, Dict[str, pd.Timestamp]] = {}

        logger.info(
            "TrendPullbackStrategy initialized | ema_trend=%d ema_pullback=%d "
            "rsi_oversold=%.1f rsi_overbought=%.1f use_rsi=%s "
            "atr_sl=%.2f atr_tp=%.2f use_dyn_stops=%s vol_spike=%.2f use_vol=%s "
            "adx_thresh=%.1f use_adx=%s cooldown=%d min_atr=%.3f%%",
            ema_trend, ema_pullback, rsi_oversold, rsi_overbought, use_rsi_confirmation,
            atr_multiplier_sl, atr_multiplier_tp, use_dynamic_atr_stops,
            volume_spike_threshold, use_volume_confirmation,
            adx_threshold, use_adx_filter, cooldown_candles, min_atr_pct,
        )

    def record_stop_out(self, symbol: str, side: str, timestamp: pd.Timestamp) -> None:
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
        return {
            symbol: {side: ts.isoformat() for side, ts in sides.items()}
            for symbol, sides in self._last_stop_out.items()
        }

    def restore_cooldown_state(self, state: dict) -> None:
        restored: Dict[str, Dict[str, pd.Timestamp]] = {}
        try:
            for symbol, sides in state.items():
                restored[symbol] = {
                    side: pd.Timestamp(ts) for side, ts in sides.items()
                }
            self._last_stop_out = restored
            logger.info("Cooldown state restored | %d active cooldown(s)", len(restored))
        except (AttributeError, ValueError, TypeError) as exc:
            logger.error("Failed to restore cooldown state (%s)", exc)

    @staticmethod
    def _normalize_side(side: str) -> str:
        upper = side.upper()
        if upper in ("BUY", "LONG"):
            return "LONG"
        if upper in ("SELL", "SHORT"):
            return "SHORT"
        raise ValueError(f"invalid side: {side}")

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

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise InvalidDataFrameError(f"Missing required columns: {missing}")

    def _min_required_rows(self) -> int:
        return max(self.ema_trend, self.rsi_period, self.atr_period, self.adx_period * 2, self.volume_ma_period) + 5

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
        return adx.where(smoothed_tr.notna() & dx.notna(), other=pd.NA).astype("float64")

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        import numpy as np

        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(avg_loss != 0, other=100.0)
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), other=50.0)
        return rsi.where(avg_gain.notna() & avg_loss.notna(), other=pd.NA).astype("float64")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df)
        out = df.copy().reset_index(drop=True)
        out["ema_trend"] = self._ema(out["close"], self.ema_trend)
        out["ema_pullback"] = self._ema(out["close"], self.ema_pullback)
        out["rsi"] = self._rsi(out["close"], self.rsi_period)
        out["atr"] = self._atr(out, self.atr_period)
        out["adx"] = self._adx(out, self.adx_period)
        out["vol_ma"] = out["volume"].rolling(window=self.volume_ma_period).mean()
        return out

    def analyze(self, symbol: str, df: pd.DataFrame) -> Signal:
        self._validate_columns(df)

        min_rows = self._min_required_rows()
        if len(df) < min_rows:
            raise InsufficientDataError(f"Need at least {min_rows} rows, got {len(df)}")

        indicators = self.calculate_indicators(df)
        last_closed = indicators.iloc[-2]
        prev_closed = indicators.iloc[-3]

        required_fields = ["ema_trend", "ema_pullback", "rsi", "atr", "adx", "vol_ma"]
        if last_closed[required_fields].isna().any() or prev_closed[required_fields].isna().any():
            raise InsufficientDataError("Indicators contain NaN values on evaluated candles")

        close_now = float(last_closed["close"])
        ema_trend_now = float(last_closed["ema_trend"])
        ema_pullback_now = float(last_closed["ema_pullback"])
        ema_pullback_prev = float(prev_closed["ema_pullback"])
        close_prev = float(prev_closed["close"])
        rsi_prev = float(prev_closed["rsi"])
        adx_now = float(last_closed["adx"])
        volume_now = float(last_closed["volume"])
        vol_ma_now = float(last_closed["vol_ma"])

        macro_uptrend = close_now > ema_trend_now
        macro_downtrend = close_now < ema_trend_now

        # Enhanced trigger: allow touch or close near pullback EMA
        pulled_back_to_ema = close_prev <= ema_pullback_prev * 1.002
        pulled_back_from_ema = close_prev >= ema_pullback_prev * 0.998
        closed_back_above_ema = close_now > ema_pullback_now
        closed_back_below_ema = close_now < ema_pullback_now

        long_setup = macro_uptrend and pulled_back_to_ema and closed_back_above_ema
        short_setup = macro_downtrend and pulled_back_from_ema and closed_back_below_ema

        if self.use_rsi_confirmation:
            long_setup = long_setup and (rsi_prev <= self.rsi_oversold)
            short_setup = short_setup and (rsi_prev >= self.rsi_overbought)

        if self.use_volume_confirmation:
            if vol_ma_now > 0:
                volume_ok = volume_now >= (vol_ma_now * self.volume_spike_threshold)
                long_setup = long_setup and volume_ok
                short_setup = short_setup and volume_ok
            elif long_setup or short_setup:
                # vol_ma_now <= 0 means we can't actually evaluate the
                # volume-spike condition (missing/zero volume data from
                # the exchange — not unusual on thinner alts during quiet
                # hours). Silently skipping the filter here would let a
                # setup through UNCONFIRMED by volume without any trace in
                # the logs — exactly the kind of silent failure a
                # production audit needs to catch. Log it explicitly and
                # fail safe (suppress the signal) rather than silently
                # bypassing a configured risk control.
                logger.warning(
                    "Volume confirmation could not be evaluated for this candle "
                    "(vol_ma=%.4f <= 0, likely missing/zero volume data) — "
                    "suppressing the %s setup rather than silently letting it "
                    "through unconfirmed.",
                    vol_ma_now, "long" if long_setup else "short",
                )
                long_setup = False
                short_setup = False

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
        last_closed_ts = pd.Timestamp(candle_timestamp)
        try:
            candle_interval: Optional[pd.Timedelta] = last_closed_ts - pd.Timestamp(prev_timestamp)
        except Exception as exc:  # noqa: BLE001
            # Falls back to a safe default (5 minutes, used by
            # _cooldown_active when interval is None) — not fatal, but
            # worth a trace-level breadcrumb rather than being fully
            # invisible, since a malformed timestamp is itself a signal
            # something upstream (candle data) may be off.
            logger.debug("Could not compute candle interval from timestamps: %s", exc)
            candle_interval = None

        side: SignalSide = raw_side
        reason: Optional[str] = None

        if side != "HOLD" and self.use_adx_filter and adx_now < self.adx_threshold:
            reason = "weak_trend_adx"
            side = "HOLD"

        if side != "HOLD" and atr_pct < self.min_atr_pct:
            reason = "low_volatility"
            side = "HOLD"

        if side != "HOLD" and self._cooldown_active(symbol, side, last_closed_ts, candle_interval):
            reason = "cooldown_active"
            side = "HOLD"

        # Dynamic ATR Stop-Loss & Take-Profit
        if self.use_dynamic_atr_stops:
            sl_distance = atr_value * self.atr_multiplier_sl
            tp_distance = atr_value * self.atr_multiplier_tp
        else:
            sl_distance = atr_value * 1.5
            tp_distance = sl_distance * self.risk_reward_ratio

        if self.tp_atr_multiplier is not None:
            tp_distance = atr_value * self.tp_atr_multiplier

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
            timestamp=last_closed_ts.isoformat(),
            reason=reason,
        )

        return signal