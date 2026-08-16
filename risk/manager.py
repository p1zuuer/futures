"""
risk/manager.py

Position-sizing risk manager. Given an account equity, a desired risk
percentage per trade, an entry price, and a stop-loss price, calculates a
safe order quantity and leverage that:

    1. Risks no more than `risk_per_trade_pct` of equity if SL is hit.
    2. Never exceeds `max_position_leverage`.
    3. Never requires more margin than the account's free equity.
    4. Respects an exchange minimum order notional.

Also enforces a daily max-drawdown circuit breaker: once realized losses
for the current UTC day reach `max_daily_loss_pct` of the day's starting
equity, `calculate_position()` rejects every new trade until the next UTC
day rolls over — a hard backstop against exactly the kind of chained-loss
"death by a thousand cuts" that repeated stop-outs in choppy conditions
can cause.

Author: Senior Python Async Developer & Systems Architect
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger("risk_manager")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


@dataclass
class PositionPlan:
    """Result of a position-sizing calculation."""

    valid: bool
    symbol: str
    side: str
    quantity: float
    leverage: float
    entry_price: float
    stop_loss: float
    take_profit: float
    margin_required: float
    risk_amount_usd: float
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": round(self.quantity, 8),
            "leverage": round(self.leverage, 4),
            "entry_price": round(self.entry_price, 6),
            "stop_loss": round(self.stop_loss, 6),
            "take_profit": round(self.take_profit, 6),
            "margin_required": round(self.margin_required, 6),
            "risk_amount_usd": round(self.risk_amount_usd, 6),
            "rejection_reason": self.rejection_reason,
        }


class RiskManager:
    """
    Calculates trade position size from equity risk, honoring a max
    leverage cap and exchange minimum order notional. Also enforces a
    daily max-drawdown circuit breaker (see module docstring).
    """

    MIN_ORDER_NOTIONAL_USD: float = 1.0

    # Reserve a small slice of free margin so the exchange's execution fee
    # (deducted on top of margin at fill time) never pushes an order over
    # the account's available balance. This must exceed the exchange's
    # taker fee rate (e.g. 0.05% on dYdX v4) with headroom for slippage.
    MARGIN_SAFETY_BUFFER_PCT: float = 0.5  # reserve 0.5% of free margin

    def __init__(
        self,
        risk_per_trade_pct: float = 1.0,
        max_position_leverage: float = 2.0,
        max_daily_loss_pct: float = 5.0,
    ) -> None:
        if risk_per_trade_pct <= 0 or risk_per_trade_pct > 100:
            raise ValueError("risk_per_trade_pct must be within (0, 100]")
        if max_position_leverage <= 0:
            raise ValueError("max_position_leverage must be positive")
        if max_daily_loss_pct <= 0 or max_daily_loss_pct > 100:
            raise ValueError("max_daily_loss_pct must be within (0, 100]")

        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_leverage = max_position_leverage
        self.max_daily_loss_pct = max_daily_loss_pct

        # Daily circuit-breaker state (UTC calendar day).
        self._current_day: Optional[date] = None
        self._daily_starting_equity: Optional[float] = None
        self._daily_realized_pnl_usd: float = 0.0
        self._kill_switch_active: bool = False

        logger.info(
            "RiskManager initialized | risk_per_trade=%.2f%% max_leverage=%.1fx "
            "max_daily_loss=%.2f%%",
            risk_per_trade_pct, max_position_leverage, max_daily_loss_pct,
        )

    # ------------------------------------------------------------------ #
    # Daily loss circuit breaker
    # ------------------------------------------------------------------ #

    def _maybe_roll_over_day(self, equity_usd: float) -> None:
        """Reset daily tracking (and clear the kill switch) when the UTC
        calendar date has changed since the last observed equity."""
        today = datetime.now(timezone.utc).date()
        if self._current_day == today:
            return

        previous_day = self._current_day
        was_active = self._kill_switch_active

        self._current_day = today
        self._daily_starting_equity = equity_usd
        self._daily_realized_pnl_usd = 0.0
        self._kill_switch_active = False

        if previous_day is not None:
            logger.info(
                "NEW TRADING DAY | %s -> %s | daily loss circuit breaker reset "
                "(was_active=%s) | starting_equity=$%.4f",
                previous_day, today, was_active, equity_usd,
            )

    def record_realized_pnl(self, pnl_usd: float, equity_usd: float) -> None:
        """
        Record realized PnL from a closed position (win or loss) and
        update the daily drawdown circuit breaker. Should be called by the
        caller (TradingBot) immediately after every position close.

        `equity_usd` is the account equity AFTER this PnL was applied —
        used both to (re-)establish the day's starting-equity baseline on
        first call of a new day, and to log current standing.
        """
        self._maybe_roll_over_day(equity_usd)
        # First call of a fresh day: use pre-PnL equity as the baseline
        # rather than post-PnL, so the drawdown % is measured against what
        # the account actually started the day with.
        if self._daily_starting_equity is None:
            self._daily_starting_equity = equity_usd - pnl_usd

        self._daily_realized_pnl_usd += pnl_usd

        baseline = self._daily_starting_equity or equity_usd
        drawdown_pct = (-self._daily_realized_pnl_usd / baseline * 100.0) if baseline > 0 else 0.0

        logger.info(
            "DAILY PNL UPDATE | realized_today=$%.4f drawdown=%.2f%% "
            "(limit=%.2f%%) equity=$%.4f",
            self._daily_realized_pnl_usd, drawdown_pct, self.max_daily_loss_pct, equity_usd,
        )

        if drawdown_pct >= self.max_daily_loss_pct and not self._kill_switch_active:
            self._kill_switch_active = True
            logger.error(
                "🛑 DAILY MAX LOSS CIRCUIT BREAKER TRIGGERED | drawdown=%.2f%% >= "
                "limit=%.2f%% | blocking all new orders until the next UTC day "
                "(current day: %s)",
                drawdown_pct, self.max_daily_loss_pct, self._current_day,
            )

    def is_daily_loss_limit_hit(self) -> bool:
        """Whether the daily circuit breaker is currently blocking new trades."""
        return self._kill_switch_active

    def get_daily_stats(self) -> dict:
        """Telemetry snapshot of the current day's PnL tracking."""
        baseline = self._daily_starting_equity
        drawdown_pct = (
            (-self._daily_realized_pnl_usd / baseline * 100.0)
            if baseline and baseline > 0
            else 0.0
        )
        return {
            "trading_day": str(self._current_day) if self._current_day else None,
            "daily_starting_equity": round(baseline, 4) if baseline is not None else None,
            "daily_realized_pnl_usd": round(self._daily_realized_pnl_usd, 4),
            "daily_drawdown_pct": round(drawdown_pct, 4),
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "kill_switch_active": self._kill_switch_active,
        }

    def reset_daily_circuit_breaker(self) -> None:
        """Manually clear the kill switch without waiting for the day to
        roll over. Mainly useful for tests or an explicit operator override."""
        self._kill_switch_active = False
        logger.warning("Daily loss circuit breaker manually reset.")

    def calculate_position(
        self,
        symbol: str,
        side: str,
        equity_usd: float,
        free_margin_usd: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> PositionPlan:
        """
        Compute a risk-bounded position size.

        The trade risks `risk_per_trade_pct` of `equity_usd` if the stop
        loss is hit. Quantity is derived from that risk budget divided by
        the per-unit SL distance, then leverage is chosen as the minimum
        needed to support that notional within `max_position_leverage` and
        the account's `free_margin_usd`.

        Rejects immediately (without any sizing math) if the daily max
        loss circuit breaker is currently active for the current UTC day.
        """
        self._maybe_roll_over_day(equity_usd)

        if self._kill_switch_active:
            stats = self.get_daily_stats()
            return self._reject(
                symbol, side, entry_price, stop_loss, take_profit,
                f"daily max loss circuit breaker active: drawdown="
                f"{stats['daily_drawdown_pct']:.2f}% >= limit="
                f"{self.max_daily_loss_pct:.2f}% — blocked until next UTC day",
            )

        sl_distance = abs(entry_price - stop_loss)

        if sl_distance <= 0:
            return self._reject(
                symbol, side, entry_price, stop_loss, take_profit,
                "stop_loss distance must be non-zero",
            )
        if equity_usd <= 0:
            return self._reject(
                symbol, side, entry_price, stop_loss, take_profit,
                "account equity is zero or negative",
            )
        if free_margin_usd <= 0:
            return self._reject(
                symbol, side, entry_price, stop_loss, take_profit,
                "no free margin available",
            )

        # Leave headroom for the exchange's execution fee (deducted from
        # balance on top of margin at fill time), so a fully-utilized
        # account doesn't get its order rejected for insufficient funds.
        usable_free_margin = free_margin_usd * (1.0 - self.MARGIN_SAFETY_BUFFER_PCT / 100.0)

        risk_amount_usd = equity_usd * (self.risk_per_trade_pct / 100.0)

        # Quantity such that a full SL hit loses exactly risk_amount_usd.
        raw_quantity = risk_amount_usd / sl_distance
        notional = raw_quantity * entry_price

        if notional < self.MIN_ORDER_NOTIONAL_USD:
            # Scale up to the exchange minimum notional if risk sizing
            # alone would produce a dust order; this increases risk beyond
            # the target pct but stays a valid, executable trade.
            raw_quantity = self.MIN_ORDER_NOTIONAL_USD / entry_price
            notional = raw_quantity * entry_price

        # Minimum leverage required so that required margin fits within the
        # usable free margin (after fee buffer), capped at max_position_leverage.
        min_leverage_for_margin = (
            notional / usable_free_margin if usable_free_margin > 0 else float("inf")
        )
        leverage = max(1.0, min_leverage_for_margin)
        leverage = min(leverage, self.max_position_leverage)

        margin_required = notional / leverage

        if margin_required > usable_free_margin + 1e-9:
            # Even at max leverage, this order does not fit — shrink quantity
            # to fit exactly within usable free margin at max leverage.
            max_notional = usable_free_margin * self.max_position_leverage
            if max_notional < self.MIN_ORDER_NOTIONAL_USD:
                return self._reject(
                    symbol, side, entry_price, stop_loss, take_profit,
                    f"free margin ${free_margin_usd:.4f} too small to open "
                    f"a ${self.MIN_ORDER_NOTIONAL_USD:.2f} minimum order even "
                    f"at {self.max_position_leverage:.1f}x leverage",
                )
            raw_quantity = max_notional / entry_price
            notional = raw_quantity * entry_price
            leverage = self.max_position_leverage
            margin_required = notional / leverage

        actual_risk_usd = raw_quantity * sl_distance

        plan = PositionPlan(
            valid=True,
            symbol=symbol,
            side=side,
            quantity=raw_quantity,
            leverage=leverage,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin_required=margin_required,
            risk_amount_usd=actual_risk_usd,
        )

        logger.info(
            "POSITION PLAN | %s %s qty=%.8f lev=%.2fx margin=$%.4f risk=$%.4f",
            symbol, side, plan.quantity, plan.leverage,
            plan.margin_required, plan.risk_amount_usd,
        )
        return plan

    def _reject(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        reason: str,
    ) -> PositionPlan:
        logger.warning("POSITION PLAN REJECTED | %s %s reason=%s", symbol, side, reason)
        return PositionPlan(
            valid=False,
            symbol=symbol,
            side=side,
            quantity=0.0,
            leverage=0.0,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin_required=0.0,
            risk_amount_usd=0.0,
            rejection_reason=reason,
        )


if __name__ == "__main__":
    rm = RiskManager(risk_per_trade_pct=1.0, max_position_leverage=2.0, max_daily_loss_pct=5.0)
    plan = rm.calculate_position(
        symbol="ETH-USD",
        side="BUY",
        equity_usd=15.0,
        free_margin_usd=15.0,
        entry_price=3080.0,
        stop_loss=3049.92,
        take_profit=3140.15,
    )
    print(plan.to_dict())
    print(rm.get_daily_stats())