"""
risk/kill_switch.py

Hard-Kill-Switch system for Canary Live Trading. Enforces strict safety limits
on daily drawdown, consecutive losses, position sizing, execution slippage, and order rates.
"""

from __future__ import annotations

import logging
import time
from typing import List

logger = logging.getLogger("kill_switch")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class KillSwitchTriggered(Exception):
    """Critical safety halt exception caught at the top-level loop in main.py."""
    def __init__(self, reason: str, check_name: str):
        self.reason = reason
        self.check_name = check_name
        super().__init__(reason)


class KillSwitch:
    def __init__(self, config):
        self.max_daily_loss_pct = config.KILL_MAX_DAILY_LOSS_PCT
        self.max_consecutive_losses = config.KILL_MAX_CONSECUTIVE_LOSSES
        self.max_position_notional_pct = config.KILL_MAX_POSITION_NOTIONAL_PCT
        self.max_slippage_pct = config.KILL_MAX_SLIPPAGE_PCT
        self.max_orders_per_hour = config.KILL_MAX_ORDERS_PER_HOUR
        self.heartbeat_timeout_sec = config.KILL_HEARTBEAT_TIMEOUT_SEC
        self._order_timestamps: List[float] = []

    def check_daily_loss(self, current_equity: float, day_start_equity: float):
        if day_start_equity <= 0:
            return
        daily_loss_pct = (day_start_equity - current_equity) / day_start_equity * 100.0
        if daily_loss_pct >= self.max_daily_loss_pct:
            reason = f"Daily loss of {daily_loss_pct:.2f}% exceeds hard limit of {self.max_daily_loss_pct}%."
            logger.critical(reason)
            raise KillSwitchTriggered(reason=reason, check_name="DAILY_LOSS_LIMIT")

    def check_consecutive_losses(self, trade_history: list):
        if not trade_history:
            return
        consecutive = 0
        for trade in reversed(trade_history):
            # check return_pct
            ret = getattr(trade, "return_pct", None)
            if ret is None and isinstance(trade, dict):
                ret = trade.get("return_pct", 0.0)
            if ret is not None and ret <= 0:
                consecutive += 1
            else:
                break

        if consecutive >= self.max_consecutive_losses:
            reason = f"Detected {consecutive} consecutive losses (limit: {self.max_consecutive_losses})."
            logger.critical(reason)
            raise KillSwitchTriggered(reason=reason, check_name="CONSECUTIVE_LOSSES")

    def check_position_size(self, notional: float, account_equity: float):
        if account_equity <= 0:
            return
        pos_pct = (notional / account_equity) * 100.0
        if pos_pct > self.max_position_notional_pct:
            reason = f"Position notional {pos_pct:.2f}% of equity exceeds limit of {self.max_position_notional_pct}%."
            logger.critical(reason)
            raise KillSwitchTriggered(reason=reason, check_name="POSITION_SIZE_LIMIT")

    def check_execution_slippage(self, expected_price: float, actual_fill_price: float, side: str):
        if expected_price <= 0:
            return
        if side.upper() == "BUY":
            slippage = (actual_fill_price - expected_price) / expected_price * 100.0
        else:
            slippage = (expected_price - actual_fill_price) / expected_price * 100.0

        if slippage > self.max_slippage_pct:
            reason = f"Execution slippage of {slippage:.3f}% exceeds limit of {self.max_slippage_pct}%."
            logger.critical(reason)
            raise KillSwitchTriggered(reason=reason, check_name="SLIPPAGE_LIMIT")

    def check_order_rate(self):
        now = time.time()
        # Clean rolling window older than 1 hour
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 3600]
        self._order_timestamps.append(now)

        if len(self._order_timestamps) > self.max_orders_per_hour:
            reason = f"Order rate {len(self._order_timestamps)} orders/hour exceeds limit of {self.max_orders_per_hour}."
            logger.critical(reason)
            raise KillSwitchTriggered(reason=reason, check_name="ORDER_RATE_LIMIT")
