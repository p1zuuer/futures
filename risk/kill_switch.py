"""
risk/kill_switch.py

Hard-Kill-Switch system for Canary Live Trading. Enforces strict safety limits
on daily drawdown, consecutive losses, position sizing, execution slippage, and order rates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, Any

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
    def __init__(self, config, state_store: Optional[Any] = None, notifier: Optional[Any] = None):
        self.max_daily_loss_pct = getattr(config, "kill_max_daily_loss_pct", None)
        if self.max_daily_loss_pct is None:
            self.max_daily_loss_pct = getattr(config, "KILL_MAX_DAILY_LOSS_PCT", 2.0)

        self.max_consecutive_losses = getattr(config, "kill_max_consecutive_losses", None)
        if self.max_consecutive_losses is None:
            self.max_consecutive_losses = getattr(config, "KILL_MAX_CONSECUTIVE_LOSSES", 3)

        self.max_position_notional_pct = getattr(config, "kill_max_position_notional_pct", None)
        if self.max_position_notional_pct is None:
            self.max_position_notional_pct = getattr(config, "KILL_MAX_POSITION_NOTIONAL_PCT", 5.0)

        self.max_slippage_pct = getattr(config, "kill_max_slippage_pct", None)
        if self.max_slippage_pct is None:
            self.max_slippage_pct = getattr(config, "KILL_MAX_SLIPPAGE_PCT", 0.5)

        self.max_orders_per_hour = getattr(config, "kill_max_orders_per_hour", None)
        if self.max_orders_per_hour is None:
            self.max_orders_per_hour = getattr(config, "KILL_MAX_ORDERS_PER_HOUR", 10)

        self.heartbeat_timeout_sec = getattr(config, "kill_heartbeat_timeout_sec", None)
        if self.heartbeat_timeout_sec is None:
            self.heartbeat_timeout_sec = getattr(config, "KILL_HEARTBEAT_TIMEOUT_SEC", 300.0)

        self._order_timestamps: List[float] = []
        
        self.state_store = state_store
        self.notifier = notifier
        self.is_active: bool = False
        self.reason: Optional[str] = None
        self.check_name: Optional[str] = None

        # Restore state if available
        if self.state_store is not None:
            saved_state = self.state_store.load()
            if saved_state is not None:
                self.is_active = bool(saved_state.get("kill_switch_active", False))
                self.reason = saved_state.get("kill_switch_reason")
                self.check_name = saved_state.get("kill_switch_check_name")
                if self.is_active:
                    logger.critical(
                        "🛑 KillSwitch RESTORED AS ACTIVE from state store! Reason: %s (check: %s)",
                        self.reason, self.check_name,
                    )

    def _persist(self) -> None:
        if self.state_store is not None:
            try:
                current_state = self.state_store.load() or {}
                current_state.update({
                    "kill_switch_active": self.is_active,
                    "kill_switch_reason": self.reason,
                    "kill_switch_check_name": self.check_name,
                })
                self.state_store.save(current_state)
            except Exception as exc:
                logger.error("Failed to persist kill switch state: %s", exc)

    async def _trigger(self, reason: str, check_name: str) -> None:
        self.is_active = True
        self.reason = reason
        self.check_name = check_name
        logger.critical(reason)
        self._persist()

        if self.notifier is not None:
            try:
                send_method = getattr(self.notifier, "send_error_alert", None) or getattr(self.notifier, "_send", None)
                if callable(send_method):
                    alert_text = f"🛑 *KILL SWITCH TRIGGERED* 🛑\nCheck: `{check_name}`\nReason: {reason}"
                    try:
                        await asyncio.wait_for(send_method(alert_text), timeout=10.0)
                    except Exception as e:
                        logger.error("Failed to send critical alert: %s", e)
            except Exception as e:
                logger.error("Failed to send Telegram alert for KillSwitch trigger: %s", e)

        raise KillSwitchTriggered(reason=reason, check_name=check_name)

    async def check_daily_loss(self, current_equity: float, day_start_equity: float):
        if self.is_active:
            raise KillSwitchTriggered(reason=self.reason or "Kill switch is active", check_name=self.check_name or "ACTIVE")
        if day_start_equity <= 0:
            return
        daily_loss_pct = (day_start_equity - current_equity) / day_start_equity * 100.0
        if daily_loss_pct >= self.max_daily_loss_pct:
            reason = f"Daily loss of {daily_loss_pct:.2f}% exceeds hard limit of {self.max_daily_loss_pct}%."
            await self._trigger(reason=reason, check_name="DAILY_LOSS_LIMIT")

    async def check_consecutive_losses(self, trade_history: list):
        if self.is_active:
            raise KillSwitchTriggered(reason=self.reason or "Kill switch is active", check_name=self.check_name or "ACTIVE")
        if not trade_history:
            return
        consecutive = 0
        for trade in reversed(trade_history):
            ret = getattr(trade, "return_pct", None)
            if ret is None and isinstance(trade, dict):
                ret = trade.get("return_pct", 0.0)
            if ret is not None and ret <= 0:
                consecutive += 1
            else:
                break

        if consecutive >= self.max_consecutive_losses:
            reason = f"Detected {consecutive} consecutive losses (limit: {self.max_consecutive_losses})."
            await self._trigger(reason=reason, check_name="CONSECUTIVE_LOSSES")

    async def check_position_size(self, notional: float, account_equity: float):
        if self.is_active:
            raise KillSwitchTriggered(reason=self.reason or "Kill switch is active", check_name=self.check_name or "ACTIVE")
        if account_equity <= 0:
            return
        pos_pct = (notional / account_equity) * 100.0
        if pos_pct > self.max_position_notional_pct:
            reason = f"Position notional {pos_pct:.2f}% of equity exceeds limit of {self.max_position_notional_pct}%."
            await self._trigger(reason=reason, check_name="POSITION_SIZE_LIMIT")

    async def check_execution_slippage(self, expected_price: float, actual_fill_price: float, side: str):
        if self.is_active:
            raise KillSwitchTriggered(reason=self.reason or "Kill switch is active", check_name=self.check_name or "ACTIVE")
        if expected_price <= 0:
            return
        if side.upper() == "BUY":
            slippage = (actual_fill_price - expected_price) / expected_price * 100.0
        else:
            slippage = (expected_price - actual_fill_price) / expected_price * 100.0

        if slippage > self.max_slippage_pct:
            reason = f"Execution slippage of {slippage:.3f}% exceeds limit of {self.max_slippage_pct}%."
            await self._trigger(reason=reason, check_name="SLIPPAGE_LIMIT")

    async def check_order_rate(self):
        if self.is_active:
            raise KillSwitchTriggered(reason=self.reason or "Kill switch is active", check_name=self.check_name or "ACTIVE")
        now = time.time()
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 3600]
        self._order_timestamps.append(now)

        if len(self._order_timestamps) > self.max_orders_per_hour:
            reason = f"Order rate {len(self._order_timestamps)} orders/hour exceeds limit of {self.max_orders_per_hour}."
            await self._trigger(reason=reason, check_name="ORDER_RATE_LIMIT")
