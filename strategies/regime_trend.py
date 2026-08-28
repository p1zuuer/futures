"""
strategies/regime_trend.py

Regime-Gated Trend Following Strategy (Hypothesis #2).
Applies ADX-based regime gating (adx > 22 for last 5 candles) and
dynamic ATR-based stop-loss, take-profit, max hold, and regime invalidation exits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

import pandas as pd

logger = logging.getLogger("regime_trend_strategy")
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


@dataclass
class RegimeSignal:
    symbol: str
    side: SignalSide
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    timestamp: str
    reason: Optional[str] = None

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


class RegimeTrendStrategy:
    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 100,
        adx_period: int = 14,
        adx_min: float = 22.0,
        adx_lookback_bars: int = 5,
        atr_period: int = 14,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
        max_hold_bars: int = 72,
        cooldown_bars: int = 6,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_min = adx_min
        self.adx_lookback_bars = adx_lookback_bars
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_hold_bars = max_hold_bars
        self.cooldown_bars = cooldown_bars
        self._last_stop_out: Dict[str, Dict[str, pd.Timestamp]] = {}

    def get_cooldown_state(self) -> dict:
        return {
            symbol: {side: ts.isoformat() for side, ts in sides.items()}
            for symbol, sides in self._last_stop_out.items()
        }

    def load_cooldown_state(self, state: dict) -> None:
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

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # EMAs
        df["ema_fast"] = close.ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.ema_slow, adjust=False).mean()
        df["ema_slow_slope"] = df["ema_slow"].diff()

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=self.atr_period).mean()

        # ADX (Wilder's smoothing)
        plus_dm = high.diff()
        minus_dm = low.shift() - low
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr_smooth = tr.ewm(alpha=1/self.adx_period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / tr_smooth)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        df["adx"] = dx.ewm(alpha=1/self.adx_period, adjust=False).mean()

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise InvalidDataFrameError(f"DataFrame is missing required columns: {missing}")

    def check_regime_invalidation(self, symbol: str, df: pd.DataFrame, position_side: str) -> bool:
        """
        Check if the trend regime has invalidated for an open position
        (e.g. slow EMA slope inverted against the position direction across
        the last 2 closed bars).
        """
        if len(df) < self.ema_slow + 5:
            return False
        indicators = self.compute_indicators(df)
        last_two = indicators.iloc[-3:-1]
        slopes = last_two["ema_slow_slope"]

        if position_side.upper() == "LONG" and (slopes < 0).all():
            return True
        if position_side.upper() == "SHORT" and (slopes > 0).all():
            return True
        return False

    def analyze(self, symbol: str, df: pd.DataFrame) -> RegimeSignal:
        self._validate_columns(df)
        return self.generate_signal(df, symbol)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> RegimeSignal:
        if len(df) < max(self.ema_slow, self.adx_period * 2, self.atr_period) + 10:
            return RegimeSignal(symbol, "HOLD", 0, 0, 0, 0, str(df.iloc[-1]["timestamp"]), reason="insufficient_data")

        indicators = self.compute_indicators(df)
        
        # Evaluate strictly on PREV closed candles (-2 and -3) to prevent look-ahead bias
        prev_closed = indicators.iloc[-2]
        last_closed = indicators.iloc[-3]

        if any(pd.isna(prev_closed[col]) for col in ["ema_fast", "ema_slow", "atr", "adx", "ema_slow_slope"]):
            return RegimeSignal(symbol, "HOLD", 0, 0, 0, 0, str(prev_closed["timestamp"]), reason="nan_indicators")

        # Regime Gate Check: ADX > adx_min for last N closed bars
        adx_window = indicators["adx"].iloc[-(self.adx_lookback_bars + 1):-1]
        trading_enabled = (adx_window > self.adx_min).all()

        if not trading_enabled:
            return RegimeSignal(symbol, "HOLD", float(prev_closed["close"]), 0, 0, float(prev_closed["atr"]), str(prev_closed["timestamp"]), reason="regime_gate_inactive")

        ema_fast_prev = float(prev_closed["ema_fast"])
        ema_slow_prev = float(prev_closed["ema_slow"])
        ema_fast_last = float(last_closed["ema_fast"])
        ema_slow_last = float(last_closed["ema_slow"])
        slope = float(prev_closed["ema_slow_slope"])
        close = float(prev_closed["close"])
        atr = float(prev_closed["atr"])

        # Crossover checks
        cross_up = (ema_fast_last <= ema_slow_last) and (ema_fast_prev > ema_slow_prev)
        cross_down = (ema_fast_last >= ema_slow_last) and (ema_fast_prev < ema_slow_prev)

        side: SignalSide = "HOLD"
        reason = "no_signal"

        if cross_up and slope > 0:
            side = "BUY"
            reason = "ema_cross_up_bullish_regime"
        elif cross_down and slope < 0:
            side = "SELL"
            reason = "ema_cross_down_bearish_regime"
        elif ema_fast_prev > ema_slow_prev and slope > 0 and close <= ema_fast_prev * 1.002:
            # Pullback entry in strong uptrend
            side = "BUY"
            reason = "pullback_bullish_regime"
        elif ema_fast_prev < ema_slow_prev and slope < 0 and close >= ema_fast_prev * 0.998:
            # Pullback entry in strong downtrend
            side = "SELL"
            reason = "pullback_bearish_regime"

        sl = close - self.atr_sl_mult * atr if side == "BUY" else close + self.atr_sl_mult * atr
        tp = close + self.atr_tp_mult * atr if side == "BUY" else close - self.atr_tp_mult * atr

        return RegimeSignal(
            symbol=symbol,
            side=side,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            atr_value=atr,
            timestamp=str(prev_closed["timestamp"]),
            reason=reason,
        )
