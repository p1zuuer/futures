"""
strategies/volatility_expansion.py

Volatility Expansion Breakout strategy: trades volatility breakouts following a
period of contraction (Bollinger Band compression + Donchian channel breakout
+ volume confirmation + rising ADX).

Indicators (calculated on 1H candles):
    - donchian_high = high.rolling(N_DONCHIAN).max()
    - donchian_low = low.rolling(N_DONCHIAN).min()
    - bb_middle = close.rolling(N_BB).mean()
    - bb_std = close.rolling(N_BB).std()
    - bb_upper = bb_middle + BB_MULT * bb_std
    - bb_lower = bb_middle - BB_MULT * bb_std
    - bb_width = (bb_upper - bb_lower) / bb_middle
    - bb_width_percentile = percentile of bb_width[t] relative to N_PERCENTILE_LOOKBACK
    - adx = ADX(high, low, close, ADX_PERIOD)
    - volume_ma = volume.rolling(N_VOL_MA).mean()
    - atr = ATR(high, low, close, ATR_PERIOD)

Signal Rules (strictly evaluated on closed candles to avoid lookahead bias):
    LONG:
        - compression_flag = bb_width_percentile[t-1] <= COMPRESSION_PERCENTILE_THRESHOLD
        - breakout_flag = close[t] > donchian_high[t-1]
        - volume_flag = volume[t] > VOLUME_CONFIRM_MULT * volume_ma[t-1]
        - adx_flag = adx[t] > ADX_MIN_FOR_ENTRY and adx[t] > adx[t-1]
        LONG_SIGNAL = compression_flag and breakout_flag and volume_flag and adx_flag

    SHORT:
        - compression_flag = bb_width_percentile[t-1] <= COMPRESSION_PERCENTILE_THRESHOLD
        - breakout_flag = close[t] < donchian_low[t-1]
        - volume_flag = volume[t] > VOLUME_CONFIRM_MULT * volume_ma[t-1]
        - adx_flag = adx[t] > ADX_MIN_FOR_ENTRY and adx[t] > adx[t-1]
        SHORT_SIGNAL = compression_flag and breakout_flag and volume_flag and adx_flag
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional, Literal

import pandas as pd
import numpy as np

from strategies.trend_ema import (
    REQUIRED_COLUMNS,
    InsufficientDataError,
    InvalidDataFrameError,
    Signal,
    SignalSide,
)

logger = logging.getLogger("volatility_expansion_strategy")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class VolatilityExpansionStrategy:
    """
    Volatility Expansion Breakout strategy with Donchian channels, Bollinger Bands width
    percentile compression filter, ADX momentum confirmation, and volume spike filter.
    """

    def __init__(
        self,
        n_donchian: int = 20,
        n_bb: int = 20,
        bb_mult: float = 2.0,
        n_percentile_lookback: int = 100,
        compression_percentile_threshold: float = 25.0,
        adx_period: int = 14,
        adx_min_for_entry: float = 20.0,
        n_vol_ma: int = 20,
        volume_confirm_mult: float = 1.2,
        atr_period: int = 14,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
        max_hold_bars: int = 48,
        cooldown_candles: int = 5,
        min_atr_pct: float = 0.05,
    ) -> None:
        if n_donchian <= 0 or n_bb <= 0 or atr_period <= 0 or adx_period <= 0 or n_vol_ma <= 0:
            raise ValueError("Indicator periods must be positive integers")
        if atr_sl_mult <= 0 or atr_tp_mult <= 0:
            raise ValueError("ATR multipliers must be positive")

        self.n_donchian = n_donchian
        self.n_bb = n_bb
        self.bb_mult = bb_mult
        self.n_percentile_lookback = n_percentile_lookback
        self.compression_percentile_threshold = compression_percentile_threshold
        self.adx_period = adx_period
        self.adx_min_for_entry = adx_min_for_entry
        self.n_vol_ma = n_vol_ma
        self.volume_confirm_mult = volume_confirm_mult
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_hold_bars = max_hold_bars
        self.cooldown_candles = cooldown_candles
        self.min_atr_pct = min_atr_pct

        self._last_stop_out: Dict[str, Dict[str, pd.Timestamp]] = {}

        logger.info(
            "VolatilityExpansionStrategy initialized | n_donchian=%d n_bb=%d bb_mult=%.1f "
            "n_percentile_lookback=%d compression_thresh=%.1f adx_period=%d adx_min=%.1f "
            "n_vol_ma=%d vol_mult=%.2f atr_period=%d atr_sl=%.2f atr_tp=%.2f "
            "max_hold=%d cooldown=%d min_atr=%.3f%%",
            n_donchian, n_bb, bb_mult, n_percentile_lookback, compression_percentile_threshold,
            adx_period, adx_min_for_entry, n_vol_ma, volume_confirm_mult, atr_period,
            atr_sl_mult, atr_tp_mult, max_hold_bars, cooldown_candles, min_atr_pct,
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
            interval = pd.Timedelta(hours=1)

        elapsed_candles = elapsed / interval
        return elapsed_candles < self.cooldown_candles

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise InvalidDataFrameError(f"Missing required columns: {missing}")

    def _min_required_rows(self) -> int:
        return max(
            self.n_donchian,
            self.n_bb,
            self.n_percentile_lookback,
            self.adx_period * 2,
            self.n_vol_ma,
            self.atr_period,
        ) + 10

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
        range1 = high - low
        range2 = (high - prev_close).abs()
        range3 = (low - prev_close).abs()
        true_range = pd.concat([range1, range2, range3], axis=1).max(axis=1)
        return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    def _adx(self, df: pd.DataFrame, period: int) -> pd.Series:
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

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df)
        out = df.copy().reset_index(drop=True)

        out["donchian_high"] = out["high"].rolling(window=self.n_donchian).max()
        out["donchian_low"] = out["low"].rolling(window=self.n_donchian).min()

        out["bb_middle"] = out["close"].rolling(window=self.n_bb).mean()
        out["bb_std"] = out["close"].rolling(window=self.n_bb).std()
        out["bb_upper"] = out["bb_middle"] + self.bb_mult * out["bb_std"]
        out["bb_lower"] = out["bb_middle"] - self.bb_mult * out["bb_std"]
        
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]

        # Calculate rolling percentile of bb_width over n_percentile_lookback
        def rolling_percentile(s: pd.Series, window: int, percentile: float) -> pd.Series:
            return s.rolling(window=window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0 if len(x) >= window else np.nan,
                raw=False
            )

        out["bb_width_percentile"] = rolling_percentile(out["bb_width"], self.n_percentile_lookback, self.compression_percentile_threshold)
        
        out["adx"] = self._adx(out, self.adx_period)
        out["volume_ma"] = out["volume"].rolling(window=self.n_vol_ma).mean()
        out["atr"] = self._atr(out, self.atr_period)

        return out

    def analyze(self, symbol: str, df: pd.DataFrame) -> Signal:
        self._validate_columns(df)

        min_rows = self._min_required_rows()
        if len(df) < min_rows:
            raise InsufficientDataError(f"Need at least {min_rows} rows, got {len(df)}")

        indicators = self.calculate_indicators(df)
        last_closed = indicators.iloc[-2]
        prev_closed = indicators.iloc[-3]

        required_fields = [
            "donchian_high", "donchian_low", "bb_width_percentile",
            "adx", "volume_ma", "atr"
        ]
        if last_closed[required_fields].isna().any() or prev_closed[required_fields].isna().any():
            raise InsufficientDataError("Indicators contain NaN values on evaluated candles")

        close_now = float(last_closed["close"])
        volume_now = float(last_closed["volume"])
        atr_value = float(last_closed["atr"])

        # Flags using shift-1 values to avoid look-ahead bias
        compression_flag = float(prev_closed["bb_width_percentile"]) <= self.compression_percentile_threshold
        
        breakout_high = float(prev_closed["donchian_high"])
        breakout_low = float(prev_closed["donchian_low"])
        breakout_long = close_now > breakout_high
        breakout_short = close_now < breakout_low

        volume_ma_prev = float(prev_closed["volume_ma"])
        volume_flag = volume_ma_prev > 0 and volume_now > (self.volume_confirm_mult * volume_ma_prev)

        adx_now = float(last_closed["adx"])
        adx_prev = float(prev_closed["adx"])
        adx_flag = adx_now > self.adx_min_for_entry and adx_now > adx_prev

        long_signal = compression_flag and breakout_long and volume_flag and adx_flag
        short_signal = compression_flag and breakout_short and volume_flag and adx_flag

        raw_side: SignalSide
        if long_signal:
            raw_side = "BUY"
        elif short_signal:
            raw_side = "SELL"
        else:
            raw_side = "HOLD"

        entry_price = close_now
        atr_pct = (atr_value / entry_price * 100.0) if entry_price > 0 else 0.0

        candle_timestamp = last_closed["timestamp"]
        prev_timestamp = prev_closed["timestamp"]
        last_closed_ts = pd.Timestamp(candle_timestamp)
        try:
            candle_interval: Optional[pd.Timedelta] = last_closed_ts - pd.Timestamp(prev_timestamp)
        except Exception:
            candle_interval = None

        side: SignalSide = raw_side
        reason: Optional[str] = None

        if side != "HOLD" and atr_pct < self.min_atr_pct:
            reason = "low_volatility"
            side = "HOLD"

        if side != "HOLD" and self._cooldown_active(symbol, side, last_closed_ts, candle_interval):
            reason = "cooldown_active"
            side = "HOLD"

        sl_distance = atr_value * self.atr_sl_mult
        tp_distance = atr_value * self.atr_tp_mult

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
