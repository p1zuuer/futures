"""
exchange/base.py

Abstract base interface shared by every exchange backend (PaperExchange,
DydxV4Adapter, and any future adapters). Defining this contract lets
strategy/risk/orchestration code (TradingBot, TelegramNotifier, etc.)
depend on a single interface regardless of whether trades are simulated
locally or routed to a real dYdX v4 account.

Author: Senior Python Async Crypto Developer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseExchange(ABC):
    """
    Common async interface implemented by all exchange backends.

    Method signatures intentionally mirror `exchange.paper_exchange.PaperExchange`
    so `TradingBot` can swap between simulated and live execution without
    any changes to strategy or orchestration code.
    """

    @abstractmethod
    async def get_balance(self) -> float:
        """Return the account's free/available USDC balance."""
        raise NotImplementedError

    @abstractmethod
    async def get_account_summary(self) -> dict:
        """
        Return a structured account snapshot:
            {
                "balance_usd": float,
                "equity_usd": float,
                "locked_margin_usd": float,
                "free_margin_usd": float,
                "margin_usage_pct": float,
                "open_positions": dict[str, dict],
                "pending_orders": dict[str, dict],
            }
        """
        raise NotImplementedError

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0,
    ) -> dict:
        """Place an order and return a structured result describing it."""
        raise NotImplementedError

    @abstractmethod
    async def on_market_tick(self, symbol: str, current_price: float) -> None:
        """React to a new price tick (fills, SL/TP checks, PnL refresh)."""
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by id."""
        raise NotImplementedError

    @abstractmethod
    async def close_position(self, symbol: str) -> float:
        """Close an open position at market. Returns realized PnL."""
        raise NotImplementedError