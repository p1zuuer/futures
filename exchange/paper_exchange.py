"""
exchange/paper_exchange.py

Async-first Paper Trading Engine simulating dYdX v4 Perpetual DEX behavior
for a micro-balance USDC account.

Features:
    - LIMIT (maker) and MARKET (taker) order placement
    - Fee simulation (0.02% maker / 0.05% taker, dYdX v4 defaults)
    - $1 minimum order notional
    - Max 2x leverage enforcement for micro accounts
    - Position tracking with unrealized/realized PnL
    - Stop-loss / take-profit trigger simulation on price ticks
    - Random slippage simulation on market orders
    - Structured logging of engine events

Market data (candles, order book, ticker price) is NOT simulated: it is
fetched in real time from the live dYdX v4 Indexer REST API
(https://indexer.dydx.trade by default) via `fetch_candles()`,
`fetch_orderbook()`, and `fetch_ticker_price()`. Only *execution* — order
matching, fills, fees, margin, and PnL — is simulated locally. This means
PAPER mode trades against the exact same real-time prices LIVE mode would
see, with zero execution risk.

Author: Senior Python/Crypto Backend Engineer
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #

logger = logging.getLogger("paper_exchange")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class PaperExchangeError(Exception):
    """Base exception for all paper exchange errors."""


class InsufficientFundsError(PaperExchangeError):
    """Raised when an order/position would exceed available equity or margin."""


class InvalidOrderError(PaperExchangeError):
    """Raised when an order fails basic validation (size, price, side, etc.)."""


class OrderNotFoundError(PaperExchangeError):
    """Raised when referencing an order id that does not exist or is closed."""


class PositionNotFoundError(PaperExchangeError):
    """Raised when referencing a position that does not exist for a symbol."""


class MarketDataUnavailableError(PaperExchangeError):
    """Raised when live market data (candles/orderbook/ticker) cannot be
    fetched from the dYdX v4 Indexer — e.g. a network failure or the
    Indexer connection not having been established via `connect()` yet."""


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.OPEN
    timestamp: str = field(default_factory=_now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "leverage": self.leverage,
        }


@dataclass
class Position:
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    leverage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin: float = 0.0  # USD margin locked for this position

    def notional(self, price: Optional[float] = None) -> float:
        p = price if price is not None else self.entry_price
        return self.quantity * p

    def update_unrealized_pnl(self, mark_price: float) -> float:
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (mark_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - mark_price) * self.quantity
        return self.unrealized_pnl

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "unrealized_pnl": round(self.unrealized_pnl, 6),
            "realized_pnl": round(self.realized_pnl, 6),
            "margin": round(self.margin, 6),
        }


@dataclass
class AccountState:
    balance_usd: float
    equity_usd: float = 0.0
    open_positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: Dict[str, Order] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.equity_usd == 0.0:
            self.equity_usd = self.balance_usd


# --------------------------------------------------------------------------- #
# Core engine
# --------------------------------------------------------------------------- #

class PaperExchange:
    """
    Simulates a dYdX v4-style perpetual DEX for local paper trading.

    All public methods are async to mirror a real exchange client interface,
    even though the underlying logic is CPU-bound and synchronous. This keeps
    the engine drop-in compatible with async trading bot event loops.
    """

    MIN_ORDER_NOTIONAL_USD: float = 1.0
    MAX_LEVERAGE: float = 2.0
    SLIPPAGE_PCT: float = 0.0005  # 0.05% max random slippage on MARKET orders

    def __init__(
        self,
        initial_balance: float = 15.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        indexer_url: Optional[str] = None,
    ) -> None:
        if initial_balance <= 0:
            raise InvalidOrderError("initial_balance must be positive")

        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.account = AccountState(balance_usd=initial_balance, equity_usd=initial_balance)
        self._last_prices: Dict[str, float] = {}
        self._lock = asyncio.Lock()

        # Real-time market data comes from the live dYdX v4 Indexer REST
        # API — never simulated/random. Resolved from config.settings when
        # available (keeps a single source of truth for the URL), falling
        # back to the public default so this class stays usable standalone.
        # Always validated/normalized to strictly mainnet (see
        # normalize_and_validate_indexer_url) — a testnet or malformed URL
        # here is the classic cause of prices that are both static and far
        # from the real market.
        from exchange.indexer_http import normalize_and_validate_indexer_url

        if indexer_url is not None:
            raw_url = indexer_url.strip()
        else:
            try:
                from config import settings as _settings
                raw_url = _settings.dydx_v4_indexer_url
            except ImportError:
                raw_url = "https://indexer.dydx.trade"

        allow_non_mainnet = os.environ.get(
            "DYDX_V4_ALLOW_NON_MAINNET_INDEXER", "false"
        ).strip().lower() in ("1", "true", "yes", "on")
        self.indexer_url = normalize_and_validate_indexer_url(raw_url, allow_non_mainnet)

        self._indexer = None  # type: Optional["IndexerClient"]
        self._indexer_connected = False
        self._stale_detector = None  # type: Optional["StalePriceDetector"]

        logger.info(
            "PaperExchange initialized | balance=$%.2f maker_fee=%.4f%% taker_fee=%.4f%% "
            "indexer=%s (real-time market data; execution is simulated)",
            initial_balance,
            maker_fee * 100,
            taker_fee * 100,
            self.indexer_url,
        )

    # ------------------------------------------------------------------ #
    # Market data connection (Indexer REST — real-time, never simulated)
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """
        Establish real-time market data access via the Indexer REST API.
        No wallet/mnemonic is needed since PaperExchange never signs or
        broadcasts anything — only read calls hit the network; all
        execution stays local.

        Market-data reads (`fetch_ticker_price`, `fetch_candles`,
        `fetch_orderbook`) bypass the SDK's `IndexerClient` and call the
        Indexer directly (see `exchange/indexer_http.py`) with cache-busting
        and no-cache headers, since those endpoints get polled every 1-2
        seconds and a cached response from any CDN/proxy in front of the
        Indexer would otherwise make prices look static and stale.
        """
        from exchange.dydx_v4_adapter import _with_retries
        from exchange.indexer_http import StalePriceDetector, indexer_get

        async def _do_connect() -> None:
            # Cheap read call to fail fast if the indexer is unreachable,
            # rather than discovering it on the first real trading-loop tick.
            await indexer_get(self.indexer_url, "/v4/height")

        await _with_retries(_do_connect, op_name="connect (PaperExchange indexer)")
        self._indexer_connected = True
        self._stale_detector = StalePriceDetector(max_consecutive_repeats=5)
        logger.info(
            "PaperExchange connected to live Indexer (direct HTTP, cache-busted) | %s",
            self.indexer_url,
        )

    async def close(self) -> None:
        """No persistent connection to tear down (every call opens a fresh
        HTTP client), but kept for interface parity with DydxV4Adapter so
        `TradingBot.shutdown()` can call it unconditionally."""
        self._indexer_connected = False
        logger.info("PaperExchange market-data connection closed.")

    # ------------------------------------------------------------------ #
    # State export/import — restart safety for the simulated account.
    # PaperExchange has no real backing store (unlike LIVE, where dYdX
    # itself is the source of truth for positions/balance), so a process
    # restart would otherwise silently wipe the paper account back to its
    # starting balance. These let the caller (TradingBot) persist and
    # restore the full account state across restarts.
    # ------------------------------------------------------------------ #

    def export_state(self) -> dict:
        """Serialize the full account state to a JSON-safe dict."""
        return {
            "balance_usd": self.account.balance_usd,
            "equity_usd": self.account.equity_usd,
            "open_positions": {
                symbol: pos.to_dict() for symbol, pos in self.account.open_positions.items()
            },
            "pending_orders": {
                order_id: order.to_dict()
                for order_id, order in self.account.pending_orders.items()
            },
            "last_prices": dict(self._last_prices),
        }

    def import_state(self, state: dict) -> None:
        """
        Restore account state previously produced by `export_state()`.
        Intended to be called once, immediately after construction and
        before `connect()`/any trading activity — overwrites the fresh
        `initial_balance` account created in `__init__`.
        """
        try:
            self.account.balance_usd = float(state["balance_usd"])
            self.account.equity_usd = float(state["equity_usd"])

            self.account.open_positions = {}
            for symbol, pos_dict in state.get("open_positions", {}).items():
                self.account.open_positions[symbol] = Position(
                    symbol=pos_dict["symbol"],
                    side=PositionSide(pos_dict["side"]),
                    entry_price=pos_dict["entry_price"],
                    quantity=pos_dict["quantity"],
                    leverage=pos_dict.get("leverage", 1.0),
                    stop_loss=pos_dict.get("stop_loss"),
                    take_profit=pos_dict.get("take_profit"),
                    unrealized_pnl=pos_dict.get("unrealized_pnl", 0.0),
                    realized_pnl=pos_dict.get("realized_pnl", 0.0),
                    margin=pos_dict.get("margin", 0.0),
                )

            self.account.pending_orders = {}
            for order_id, order_dict in state.get("pending_orders", {}).items():
                self.account.pending_orders[order_id] = Order(
                    id=order_dict["id"],
                    symbol=order_dict["symbol"],
                    side=OrderSide(order_dict["side"]),
                    order_type=OrderType(order_dict["order_type"]),
                    quantity=order_dict["quantity"],
                    price=order_dict.get("price"),
                    status=OrderStatus(order_dict.get("status", "OPEN")),
                    timestamp=order_dict.get("timestamp", _now()),
                    stop_loss=order_dict.get("stop_loss"),
                    take_profit=order_dict.get("take_profit"),
                    leverage=order_dict.get("leverage", 1.0),
                )

            self._last_prices = dict(state.get("last_prices", {}))

            logger.info(
                "PaperExchange state restored | balance=$%.4f equity=$%.4f "
                "open_positions=%d pending_orders=%d",
                self.account.balance_usd, self.account.equity_usd,
                len(self.account.open_positions), len(self.account.pending_orders),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.error(
                "Failed to restore PaperExchange state (%s) — continuing with a "
                "fresh account instead of a corrupted/partial one.", exc,
            )

    def _ensure_market_data_connected(self) -> None:
        if not self._indexer_connected:
            raise MarketDataUnavailableError(
                "PaperExchange.connect() must be awaited before fetching "
                "market data (candles/orderbook/ticker)."
            )

    async def fetch_candles(
        self, symbol: str, resolution: str = "1MIN", limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch real-time historical OHLCV candles directly from the live
        dYdX v4 Indexer (`GET /v4/candles/perpetualMarkets/{symbol}`),
        bypassing any intermediate cache. Returns a DataFrame shaped
        ['timestamp', 'open', 'high', 'low', 'close', 'volume'], oldest
        first — identical contract to `DydxV4Adapter.fetch_candles()` so
        strategy code is exchange-agnostic.

        `resolution` must be one of dYdX v4's supported values, e.g.
        "1MIN", "5MINS", "15MINS", "30MINS", "1HOUR", "4HOURS", "1DAY".
        """
        self._ensure_market_data_connected()

        from exchange.dydx_v4_adapter import _with_retries
        from exchange.indexer_http import indexer_get

        async def _fetch() -> dict:
            return await indexer_get(
                self.indexer_url,
                f"/v4/candles/perpetualMarkets/{symbol}",
                params={"resolution": resolution, "limit": limit},
            )

        response = await _with_retries(
            _fetch, op_name=f"get_perpetual_market_candles({symbol})",
            retry_on=(Exception,),
        )
        raw_candles = response.get("candles", [])
        if not raw_candles:
            logger.warning("No candle data returned for %s @ %s", symbol, resolution)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rows = [
            {
                "timestamp": pd.to_datetime(c["startedAt"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c.get("baseTokenVolume", 0.0)),
            }
            for c in raw_candles
        ]

        # Indexer returns candles newest-first; strategy code expects
        # oldest-first with the most recent closed candle at the end.
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        logger.debug(
            "fetch_candles(%s) latest closed candle: %s close=%.4f",
            symbol, df.iloc[-1]["timestamp"], df.iloc[-1]["close"],
        )
        return df

    async def fetch_orderbook(self, symbol: str) -> dict:
        """
        Fetch the current live order book for `symbol` directly from the
        Indexer (`GET /v4/orderbooks/perpetualMarket/{symbol}`), bypassing
        any intermediate cache. Returns the raw Indexer response shape:
            {"bids": [{"price": "...", "size": "..."}, ...],
             "asks": [{"price": "...", "size": "..."}, ...]}
        """
        self._ensure_market_data_connected()

        from exchange.dydx_v4_adapter import _with_retries
        from exchange.indexer_http import indexer_get

        async def _fetch() -> dict:
            return await indexer_get(
                self.indexer_url, f"/v4/orderbooks/perpetualMarket/{symbol}"
            )

        return await _with_retries(_fetch, op_name=f"get_perpetual_market_orderbook({symbol})")

    async def fetch_ticker_price(self, symbol: str) -> float:
        """
        Fetch the current real-time market price for `symbol` directly
        from the live dYdX v4 Indexer, bypassing the SDK's client and any
        intermediate cache (see `exchange/indexer_http.py` for why this
        matters — a cached response from a CDN/proxy is the most common
        cause of a price that looks both static and outdated). This is
        the live price PAPER mode uses to simulate fills, SL/TP triggers,
        and PnL (never a random walk, never hardcoded).

        Price source priority:
            1. Latest executed trade price
               (`GET /v4/trades/perpetualMarket/{symbol}?limit=1`) — the
               true tick-by-tick market price, updating on every fill.
            2. `oraclePrice` (fallback: `indexPrice`) from
               `GET /v4/perpetualMarkets?ticker={symbol}` if no recent
               trades are available. Note dYdX's on-chain oracle only
               updates periodically (not per-trade), so relying on it
               alone on a tight poll loop is what previously made the
               price look constant between oracle updates.
        """
        self._ensure_market_data_connected()

        from exchange.dydx_v4_adapter import ExchangeAPIError, _with_retries
        from exchange.indexer_http import indexer_get

        fetch_timestamp = datetime.now(timezone.utc).isoformat()

        async def _fetch_trades() -> dict:
            return await indexer_get(
                self.indexer_url, f"/v4/trades/perpetualMarket/{symbol}", params={"limit": 1}
            )

        price: Optional[float] = None
        source = None
        raw_price_str: Optional[str] = None

        try:
            trades_response = await _with_retries(
                _fetch_trades, op_name=f"get_perpetual_market_trades({symbol})",
                max_attempts=2,
            )
            trades = trades_response.get("trades", [])
            if trades:
                # Indexer returns trades newest-first.
                latest_trade = trades[0]
                if "price" in latest_trade:
                    raw_price_str = latest_trade["price"]
                    price = float(raw_price_str)
                    source = "trade"
        except Exception as exc:  # noqa: BLE001 - fall through to oracle price below
            logger.debug(
                "Latest-trade lookup failed for %s (%s) — falling back to oracle price.",
                symbol, exc,
            )

        if price is None:
            async def _fetch_market() -> dict:
                return await indexer_get(
                    self.indexer_url, "/v4/perpetualMarkets", params={"ticker": symbol}
                )

            response = await _with_retries(_fetch_market, op_name=f"get_perpetual_markets(price:{symbol})")
            # dYdX v4 keys the markets dict by ticker, e.g. markets["ETH-USD"].
            market = (response.get("markets", {})).get(symbol)
            if market is None:
                raise ExchangeAPIError(
                    f"Could not fetch price for {symbol}: markets['{symbol}'] not present "
                    f"in response (available keys: {list(response.get('markets', {}).keys())[:10]})"
                )

            for price_field in ("oraclePrice", "indexPrice"):
                if market.get(price_field) is not None:
                    raw_price_str = market[price_field]
                    price = float(raw_price_str)
                    source = price_field
                    break

            if price is None:
                raise ExchangeAPIError(
                    f"Could not fetch price for {symbol}: no recent trades and no "
                    f"oraclePrice/indexPrice field in market data"
                )

        if self._stale_detector is not None:
            self._stale_detector.observe(symbol, price, source or "unknown", self.indexer_url)

        # --- TEMPORARY DEBUG LOGGING ---
        # Prints the raw fetched price string (pre-float-conversion) and
        # fetch timestamp on every tick, to make it trivially verifiable
        # that fresh mainnet data is being received. Safe to downgrade to
        # logger.debug(...) or remove once live pricing is confirmed correct.
        logger.info(
            "[DEBUG-PRICE] %s ticker=%s source=%s raw='%s' parsed=%.4f indexer=%s fetched_at=%s",
            symbol, symbol, source, raw_price_str, price, self.indexer_url, fetch_timestamp,
        )

        return price

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply random slippage (0 to SLIPPAGE_PCT) unfavorable to the trader."""
        slip = random.uniform(0.0, self.SLIPPAGE_PCT)
        if side == OrderSide.BUY:
            return price * (1 + slip)
        return price * (1 - slip)

    def _required_margin(self, notional: float, leverage: float) -> float:
        return notional / leverage

    def _free_equity(self) -> float:
        """Equity not currently locked as margin in open positions."""
        locked = sum(pos.margin for pos in self.account.open_positions.values())
        return self.account.equity_usd - locked

    def _recompute_equity(self) -> None:
        unrealized_total = sum(
            pos.unrealized_pnl for pos in self.account.open_positions.values()
        )
        self.account.equity_usd = self.account.balance_usd + unrealized_total

    def _validate_new_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        leverage: float,
    ) -> None:
        if quantity <= 0:
            raise InvalidOrderError("quantity must be positive")
        if price <= 0:
            raise InvalidOrderError("price must be positive")
        if leverage <= 0 or leverage > self.MAX_LEVERAGE:
            raise InvalidOrderError(
                f"leverage must be within (0, {self.MAX_LEVERAGE}] for this account tier"
            )

        notional = quantity * price
        if notional < self.MIN_ORDER_NOTIONAL_USD:
            raise InvalidOrderError(
                f"order notional ${notional:.4f} is below the $"
                f"{self.MIN_ORDER_NOTIONAL_USD:.2f} minimum order size"
            )

        required_margin = self._required_margin(notional, leverage)
        free_equity = self._free_equity()
        if required_margin > free_equity:
            raise InsufficientFundsError(
                f"required margin ${required_margin:.4f} exceeds free equity "
                f"${free_equity:.4f}"
            )

    # ------------------------------------------------------------------ #
    # Order placement
    # ------------------------------------------------------------------ #

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
    ) -> Order:
        """
        Place a LIMIT or MARKET order.

        MARKET orders require a `price` argument representing the current
        mark/reference price (since this engine has no live order book);
        slippage is applied on top of it and the order fills immediately.

        LIMIT orders rest as pending until `on_market_tick` crosses the
        limit price.
        """
        async with self._lock:
            side_enum = OrderSide(side.upper())
            type_enum = OrderType(order_type.upper())

            if type_enum == OrderType.MARKET and price is None:
                raise InvalidOrderError("MARKET orders require a reference price")
            if type_enum == OrderType.LIMIT and price is None:
                raise InvalidOrderError("LIMIT orders require a limit price")

            self._validate_new_order(symbol, quantity, price, leverage)

            order = Order(
                id=_new_id("ord"),
                symbol=symbol,
                side=side_enum,
                order_type=type_enum,
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
            )

            logger.info(
                "ORDER PLACED | id=%s %s %s %s qty=%.6f price=%.4f lev=%.1fx",
                order.id, order.symbol, order.side.value, order.order_type.value,
                order.quantity, order.price, order.leverage,
            )

            if type_enum == OrderType.MARKET:
                await self._fill_order(order, fill_price=price, is_taker=True)
            else:
                self.account.pending_orders[order.id] = order

            return order

    async def _fill_order(self, order: Order, fill_price: float, is_taker: bool) -> None:
        """Execute a fill: apply slippage (market only), fees, open/adjust position."""
        exec_price = fill_price
        if is_taker:
            exec_price = self._apply_slippage(fill_price, order.side)

        notional = order.quantity * exec_price
        fee_rate = self.taker_fee if is_taker else self.maker_fee
        fee = notional * fee_rate
        required_margin = self._required_margin(notional, order.leverage)

        if fee + required_margin > self.account.balance_usd + 1e-9:
            order.status = OrderStatus.CANCELLED
            self.account.pending_orders.pop(order.id, None)
            raise InsufficientFundsError(
                f"insufficient balance to cover fee+margin (${fee + required_margin:.4f}) "
                f"for order {order.id}"
            )

        self.account.balance_usd -= fee
        order.status = OrderStatus.FILLED

        position_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
        existing = self.account.open_positions.get(order.symbol)

        if existing is None:
            position = Position(
                symbol=order.symbol,
                side=position_side,
                entry_price=exec_price,
                quantity=order.quantity,
                leverage=order.leverage,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                margin=required_margin,
            )
            self.account.open_positions[order.symbol] = position
        else:
            if existing.side == position_side:
                # Add to position: weighted-average entry price
                total_qty = existing.quantity + order.quantity
                existing.entry_price = (
                    existing.entry_price * existing.quantity + exec_price * order.quantity
                ) / total_qty
                existing.quantity = total_qty
                existing.margin += required_margin
                if order.stop_loss is not None:
                    existing.stop_loss = order.stop_loss
                if order.take_profit is not None:
                    existing.take_profit = order.take_profit
            else:
                # Opposing order reduces/flips the position
                await self._reduce_or_flip_position(existing, order, exec_price, required_margin)

        self.account.pending_orders.pop(order.id, None)
        self._last_prices[order.symbol] = exec_price
        self._recompute_equity()

        logger.info(
            "ORDER FILLED | id=%s %s %s qty=%.6f @ %.4f fee=$%.4f (%s)",
            order.id, order.symbol, order.side.value, order.quantity, exec_price,
            fee, "taker" if is_taker else "maker",
        )

    async def _reduce_or_flip_position(
        self,
        position: Position,
        order: Order,
        exec_price: float,
        incoming_margin: float,
    ) -> None:
        """Handle an opposing-side fill against an existing position (reduce or flip)."""
        closing_qty = min(position.quantity, order.quantity)
        if position.side == PositionSide.LONG:
            realized = (exec_price - position.entry_price) * closing_qty
        else:
            realized = (position.entry_price - exec_price) * closing_qty

        position.realized_pnl += realized
        self.account.balance_usd += realized
        released_margin = position.margin * (closing_qty / position.quantity)
        position.margin -= released_margin
        position.quantity -= closing_qty

        remainder = order.quantity - closing_qty
        if position.quantity <= 1e-12:
            # Position fully closed
            del self.account.open_positions[position.symbol]
            if remainder > 1e-12:
                # Flip: open a new position in the opposite direction
                new_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
                flipped_margin = self._required_margin(remainder * exec_price, order.leverage)
                self.account.open_positions[order.symbol] = Position(
                    symbol=order.symbol,
                    side=new_side,
                    entry_price=exec_price,
                    quantity=remainder,
                    leverage=order.leverage,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    margin=flipped_margin,
                )
        logger.info(
            "POSITION REDUCED | %s realized_pnl=$%.4f remaining_qty=%.6f",
            position.symbol, realized, max(position.quantity, 0.0),
        )

    # ------------------------------------------------------------------ #
    # Market tick handling (limit fills + SL/TP + PnL updates)
    # ------------------------------------------------------------------ #

    async def on_market_tick(self, symbol: str, current_price: float) -> None:
        """
        Process a new price tick for `symbol`:
          1. Attempt to fill any resting LIMIT orders that the price has crossed.
          2. Check open position(s) for SL/TP triggers and market-close if hit.
          3. Refresh unrealized PnL and account equity.
        """
        async with self._lock:
            self._last_prices[symbol] = current_price

            # 1. Limit order matching
            for order_id in list(self.account.pending_orders.keys()):
                order = self.account.pending_orders[order_id]
                if order.symbol != symbol or order.status != OrderStatus.OPEN:
                    continue

                crossed = (
                    order.side == OrderSide.BUY and current_price <= order.price
                ) or (
                    order.side == OrderSide.SELL and current_price >= order.price
                )
                if crossed:
                    await self._fill_order(order, fill_price=order.price, is_taker=False)

            # 2. SL / TP checks
            position = self.account.open_positions.get(symbol)
            if position is not None:
                await self._check_sl_tp(position, current_price)

            # 3. PnL / equity refresh (position may have been closed above)
            position = self.account.open_positions.get(symbol)
            if position is not None:
                position.update_unrealized_pnl(current_price)
            self._recompute_equity()

    async def _check_sl_tp(self, position: Position, current_price: float) -> None:
        triggered_reason: Optional[str] = None

        if position.side == PositionSide.LONG:
            if position.stop_loss is not None and current_price <= position.stop_loss:
                triggered_reason = "STOP_LOSS"
            elif position.take_profit is not None and current_price >= position.take_profit:
                triggered_reason = "TAKE_PROFIT"
        else:  # SHORT
            if position.stop_loss is not None and current_price >= position.stop_loss:
                triggered_reason = "STOP_LOSS"
            elif position.take_profit is not None and current_price <= position.take_profit:
                triggered_reason = "TAKE_PROFIT"

        if triggered_reason is not None:
            if triggered_reason == "STOP_LOSS":
                logger.warning(
                    "SL TRIGGERED | %s side=%s entry=%.4f trigger=%.4f",
                    position.symbol, position.side.value, position.entry_price, current_price,
                )
            else:
                logger.info(
                    "TP TRIGGERED | %s side=%s entry=%.4f trigger=%.4f",
                    position.symbol, position.side.value, position.entry_price, current_price,
                )
            await self._market_close(position, current_price)

    # ------------------------------------------------------------------ #
    # Cancel / close
    # ------------------------------------------------------------------ #

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending LIMIT order. Returns True if cancelled."""
        async with self._lock:
            order = self.account.pending_orders.get(order_id)
            if order is None or order.status != OrderStatus.OPEN:
                raise OrderNotFoundError(f"order {order_id} not found or not open")
            order.status = OrderStatus.CANCELLED
            del self.account.pending_orders[order_id]
            logger.info("ORDER CANCELLED | id=%s", order_id)
            return True

    async def close_position(self, symbol: str) -> float:
        """Market-close an open position at the last known price. Returns realized PnL."""
        async with self._lock:
            position = self.account.open_positions.get(symbol)
            if position is None:
                raise PositionNotFoundError(f"no open position for {symbol}")
            mark_price = self._last_prices.get(symbol, position.entry_price)
            return await self._market_close(position, mark_price)

    async def _market_close(self, position: Position, mark_price: float) -> float:
        """Internal: close (all of) a position at market, applying slippage + taker fee."""
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        exec_price = self._apply_slippage(mark_price, close_side)

        if position.side == PositionSide.LONG:
            realized = (exec_price - position.entry_price) * position.quantity
        else:
            realized = (position.entry_price - exec_price) * position.quantity

        notional = position.quantity * exec_price
        fee = notional * self.taker_fee

        position.realized_pnl += realized
        self.account.balance_usd += realized - fee
        self.account.balance_usd = max(self.account.balance_usd, 0.0)  # never negative

        del self.account.open_positions[position.symbol]
        self._recompute_equity()

        logger.info(
            "POSITION CLOSED | %s side=%s qty=%.6f entry=%.4f exit=%.4f "
            "realized_pnl=$%.4f fee=$%.4f new_balance=$%.4f",
            position.symbol, position.side.value, position.quantity,
            position.entry_price, exec_price, realized, fee, self.account.balance_usd,
        )
        return realized - fee

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    async def get_account_summary(self) -> dict:
        """Return balance, equity, margin usage, and per-position metrics."""
        async with self._lock:
            locked_margin = sum(pos.margin for pos in self.account.open_positions.values())
            margin_usage_pct = (
                (locked_margin / self.account.equity_usd * 100)
                if self.account.equity_usd > 0
                else 0.0
            )
            return {
                "balance_usd": round(self.account.balance_usd, 4),
                "equity_usd": round(self.account.equity_usd, 4),
                "locked_margin_usd": round(locked_margin, 4),
                "free_margin_usd": round(self._free_equity(), 4),
                "margin_usage_pct": round(margin_usage_pct, 2),
                "open_positions": {
                    sym: pos.to_dict() for sym, pos in self.account.open_positions.items()
                },
                "pending_orders": {
                    oid: o.to_dict() for oid, o in self.account.pending_orders.items()
                },
            }


# --------------------------------------------------------------------------- #
# Demo: full lifecycle simulation
# --------------------------------------------------------------------------- #

async def _demo() -> None:
    exchange = PaperExchange(initial_balance=15.0)

    print("\n=== Initial account state ===")
    print(await exchange.get_account_summary())

    # Open a LONG position on ETH at ~$3000 using MARKET order, 2x leverage,
    # with SL at $2950 and TP at $3090 (~3% target on the underlying move,
    # amplified by leverage).
    order = await exchange.place_order(
        symbol="ETH-USD",
        side="BUY",
        order_type="MARKET",
        quantity=0.0098,        # ~$29.4 notional / 2x lev = ~$14.7 margin (leaves room for fees)
        price=3000.0,
        stop_loss=2950.0,
        take_profit=3090.0,
        leverage=2.0,
    )
    print("\n=== Order after MARKET fill ===")
    print(order.to_dict())

    print("\n=== Account after opening position ===")
    print(await exchange.get_account_summary())

    # Simulate a price stream: 3000 -> 2940 (dips, would breach SL) -> 3100
    price_stream = [2995.0, 2970.0, 2940.0]
    for price in price_stream:
        print(f"\n--- Tick: ETH-USD @ {price} ---")
        await exchange.on_market_tick("ETH-USD", price)
        summary = await exchange.get_account_summary()
        print(
            f"balance=${summary['balance_usd']} equity=${summary['equity_usd']} "
            f"positions={list(summary['open_positions'].keys())}"
        )

    print("\n=== Account summary after SL breach (position should be closed) ===")
    print(await exchange.get_account_summary())

    # Re-open a fresh LONG to demonstrate a TP path on the way to $3100.
    order2 = await exchange.place_order(
        symbol="ETH-USD",
        side="BUY",
        order_type="LIMIT",
        quantity=0.005,
        price=2945.0,
        stop_loss=2900.0,
        take_profit=3010.0,
        leverage=1.5,
    )
    print("\n=== Pending LIMIT order placed ===")
    print(order2.to_dict())

    price_stream_2 = [2960.0, 2945.0, 2980.0, 3010.0, 3050.0, 3100.0]
    for price in price_stream_2:
        print(f"\n--- Tick: ETH-USD @ {price} ---")
        await exchange.on_market_tick("ETH-USD", price)
        summary = await exchange.get_account_summary()
        print(
            f"balance=${summary['balance_usd']} equity=${summary['equity_usd']} "
            f"positions={list(summary['open_positions'].keys())}"
        )

    print("\n=== Final account summary ===")
    final = await exchange.get_account_summary()
    print(final)


if __name__ == "__main__":
    logging.getLogger("paper_exchange").setLevel(logging.INFO)
    asyncio.run(_demo())