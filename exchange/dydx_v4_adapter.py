"""
exchange/dydx_v4_adapter.py

Production dYdX v4 perpetual exchange adapter, built on the official
`dydx-v4-client` Python SDK (Indexer REST client + Node gRPC client +
Cosmos-SDK wallet signing). Implements `BaseExchange` so it is a drop-in
replacement for `PaperExchange` inside `TradingBot`.

Key SDK building blocks used here (all real, from `dydx_v4_client`):
    - dydx_v4_client.indexer.rest.indexer_client.IndexerClient
        Read-only REST/HTTP access to subaccounts, positions, orders,
        candles, order books, and ticker prices. Pure httpx under the
        hood — no gRPC dependency, works identically on every platform.
    - dydx_v4_client.node.client.NodeClient
        gRPC access to the dYdX chain for signing/broadcasting
        transactions. This is the ONLY part of this adapter that depends
        on grpcio, and the only part affected by the Windows grpcio
        "unsupported platform" compatibility issue.
    - dydx_v4_client.wallet.Wallet / dydx_v4_client.key_pair.KeyPair
        Derives a Cosmos-SDK signing key from a BIP-39 mnemonic.
    - dydx_v4_client.node.market.Market
        Converts human-readable price/size into the chain's quantums/
        subticks representation for a given market, and builds signed
        Order protobuf messages (including SL/TP as conditional orders).
    - dydx_v4_client.network.NodeConfig / dydx_v4_client.indexer.candles_resolution.CandlesResolution

Connection model (Indexer/Node decoupling):
    `connect()` establishes the Indexer (REST/HTTP) connection first and
    independently of the Node (gRPC) connection. If the Node connection
    fails — including the known Windows grpcio bug — `connect()` does
    NOT raise; it logs a clear diagnostic and leaves the adapter in
    "data-only" mode. Every read method (get_balance,
    get_account_summary, fetch_candles, fetch_orderbook,
    fetch_ticker_price) only requires the Indexer and keeps working.
    Only order-placing methods (place_order, cancel_order,
    close_position) require the Node connection and raise
    `NodeConnectionError` if it's unavailable. Check `adapter.node_available`
    to know which mode you're in.

Safety: this adapter refuses to submit any order-placing transaction
unless `DYDX_V4_LIVE_TRADING_ENABLED=true` is set in the environment —
this is a hard kill-switch independent of any other configuration.

Author: Senior Python Async Crypto Developer & dYdX v4 SDK Expert
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import grpc
import pandas as pd

from exchange.base import BaseExchange

logger = logging.getLogger("dydx_v4_adapter")
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

class DydxAdapterError(Exception):
    """Base exception for all dYdX v4 adapter errors."""


class InsufficientFundsError(DydxAdapterError):
    """Raised when the subaccount does not have enough free collateral for
    the requested order (margin check performed pre-flight, mirroring the
    chain's own equity-tier / margin requirement rejection)."""


class InvalidOrderError(DydxAdapterError):
    """Raised for order validation failures (bad size, unknown market,
    unsupported order type, etc.) before a transaction is even built."""


class ConnectionNotEstablishedError(DydxAdapterError):
    """Raised when a method requiring an active connection is called
    before `connect()` has completed successfully."""


class NodeConnectionError(DydxAdapterError):
    """
    Raised when an order-placing method is called but the gRPC Node
    connection is unavailable — most commonly due to the Windows grpcio
    "unsupported platform" compatibility issue. Distinct from
    `ConnectionNotEstablishedError` so callers can tell "never connected
    at all" apart from "indexer works fine, but node/signing doesn't."
    """


class ExchangeAPIError(DydxAdapterError):
    """Wraps unexpected gRPC/HTTP failures from the Indexer or Node after
    retries have been exhausted."""


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #

async def _with_retries(
    coro_factory,
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    retry_on: tuple = (Exception,),
    op_name: str = "operation",
):
    """
    Run an async operation with exponential backoff + jitter.

    `coro_factory` is a zero-arg callable returning a fresh coroutine each
    call (coroutines can't be re-awaited, so a factory is required for
    retries to actually re-invoke the underlying call).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except retry_on as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = base_delay_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.2fs",
                op_name, attempt, max_attempts, exc, delay,
            )
            await asyncio.sleep(delay)

    raise ExchangeAPIError(f"{op_name} failed after {max_attempts} attempts: {last_exc}") from last_exc


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #

@dataclass
class _MarketMeta:
    """Cached per-symbol market metadata needed for order construction."""
    ticker: str
    clob_pair_id: int
    atomic_resolution: int
    step_base_quantums: int
    quantum_conversion_exponent: int
    subticks_per_tick: int
    raw: dict


class DydxV4Adapter(BaseExchange):
    """
    Live (and testnet) dYdX v4 perpetual exchange adapter.

    Configuration is sourced from the centralized, validated `config.settings`
    singleton when available (which correctly loads `.env` via python-dotenv
    and strictly parses booleans — see config.py), falling back to raw
    `os.environ` reads if `config` isn't importable (e.g. this module used
    standalone). Relevant variables either way:
        DYDX_V4_MNEMONIC              BIP-39 mnemonic for the trading wallet.
        DYDX_V4_NODE_URL              gRPC node endpoint (host:port, no scheme).
        DYDX_V4_INDEXER_URL           Indexer REST base URL.
        DYDX_V4_SUBACCOUNT_NUMBER     Subaccount number to trade (default 0).
        DYDX_V4_CHAIN_ID              Chain id (default "dydx-mainnet-1").
        DYDX_V4_USDC_DENOM            USDC IBC denom on the dYdX chain.
        DYDX_V4_LIVE_TRADING_ENABLED  Hard kill-switch; must be "true" to
                                       allow any order-placing transaction.
    """

    # Matches PaperExchange's minimum notional so risk sizing / signal
    # rejection behaves identically across both backends.
    MIN_ORDER_NOTIONAL_USD: float = 1.0

    def __init__(self) -> None:
        from exchange.indexer_http import normalize_and_validate_indexer_url

        allow_non_mainnet = os.environ.get(
            "DYDX_V4_ALLOW_NON_MAINNET_INDEXER", "false"
        ).strip().lower() in ("1", "true", "yes", "on")

        try:
            from config import settings as _settings

            self.mnemonic: str = _settings.dydx_v4_mnemonic
            self.node_url: str = _settings.dydx_v4_node_url
            self.indexer_url: str = normalize_and_validate_indexer_url(
                _settings.dydx_v4_indexer_url, allow_non_mainnet
            )
            self.chain_id: str = _settings.dydx_v4_chain_id
            self.usdc_denom: str = _settings.dydx_v4_usdc_denom
            self.subaccount_number: int = _settings.dydx_v4_subaccount_number
            self.live_trading_enabled: bool = _settings.dydx_v4_live_trading_enabled
            logger.debug("DydxV4Adapter configured from config.settings")
        except ImportError:
            # Standalone fallback: config.py not on the import path. Reads
            # raw os.environ directly — note this does NOT load .env files
            # (see config.py for why that matters), so this path assumes
            # the environment is already populated by the OS/shell/container.
            logger.warning(
                "config.py not importable — reading dYdX v4 settings directly "
                "from os.environ (no .env file will be loaded). Import this "
                "adapter alongside config.py for full .env support."
            )
            self.mnemonic = os.environ.get("DYDX_V4_MNEMONIC", "").strip()
            self.node_url = os.environ.get("DYDX_V4_NODE_URL", "").strip()
            self.indexer_url = normalize_and_validate_indexer_url(
                os.environ.get("DYDX_V4_INDEXER_URL", "https://indexer.dydx.trade"),
                allow_non_mainnet,
            )
            self.chain_id = os.environ.get("DYDX_V4_CHAIN_ID", "dydx-mainnet-1").strip()
            self.usdc_denom = os.environ.get(
                "DYDX_V4_USDC_DENOM",
                "ibc/8E27BA2D5493AF5636760E354E46004562C46AB7EC0CC4C1CA14E9E20E2545B5",
            ).strip()
            self.subaccount_number = int(os.environ.get("DYDX_V4_SUBACCOUNT_NUMBER", "0"))
            raw_flag = os.environ.get("DYDX_V4_LIVE_TRADING_ENABLED", "false").strip().lower()
            self.live_trading_enabled = raw_flag in ("1", "true", "yes", "on", "y", "t")

        self._indexer = None  # type: Optional["IndexerClient"]
        self._node = None  # type: Optional["NodeClient"]
        self._wallet = None  # type: Optional["Wallet"]
        self._address: Optional[str] = None
        self._market_cache: Dict[str, _MarketMeta] = {}
        self._connected = False
        self._indexer_connected = False
        self._node_connected = False
        self._stale_detector = None  # type: Optional["StalePriceDetector"]
        self._lock = asyncio.Lock()

        logger.info(
            "DydxV4Adapter initialized | chain_id=%s subaccount=%d "
            "live_trading=%s indexer=%s",
            self.chain_id, self.subaccount_number,
            "ENABLED" if self.live_trading_enabled else "DISABLED (paper-only)",
            self.indexer_url,
        )

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """
        Initialize the Indexer REST client and, separately, attempt to
        initialize the Node gRPC client + signing wallet.

        These two are intentionally decoupled: the Indexer is pure
        HTTP/HTTPS (via httpx) and works identically on every platform,
        while the Node client depends on `grpcio`, which has known
        platform-specific failures — most notably on Windows, where
        certain grpcio builds raise
        `_InactiveRpcError: StatusCode.INTERNAL "unsupported platform"`
        when opening a secure channel to a dYdX validator.

        If the Node/gRPC connection fails, this method does NOT raise —
        it logs a clear, actionable warning and leaves the adapter in
        "data-only" mode: every read method (get_balance,
        get_account_summary, fetch_candles, fetch_orderbook,
        fetch_ticker_price) keeps working normally since they only need
        the Indexer. Only order-placing methods (place_order,
        cancel_order, close_position) will raise `NodeConnectionError`
        until a working Node connection is established.
        """
        if not self.mnemonic:
            raise InvalidOrderError(
                "DYDX_V4_MNEMONIC is not set — cannot derive a signing wallet."
            )

        # --- Step 1: Indexer (REST/HTTP) — always attempted, always safe
        # on every platform including Windows.
        from dydx_v4_client.indexer.rest.indexer_client import IndexerClient

        async def _connect_indexer() -> None:
            self._indexer = IndexerClient(host=self.indexer_url)
            # Cheap read call to fail fast if the indexer URL is unreachable
            # or misconfigured, rather than discovering it on the first
            # real trading-loop call.
            await self._indexer.utility.get_height()

        await _with_retries(_connect_indexer, op_name="connect (indexer)")
        self._indexer_connected = True

        from exchange.indexer_http import StalePriceDetector
        self._stale_detector = StalePriceDetector(max_consecutive_repeats=5)

        logger.info("DydxV4Adapter Indexer connected (HTTP/HTTPS) | %s", self.indexer_url)

        # --- Step 2: Node (gRPC) — required only for signing/broadcasting
        # transactions. Failures here are isolated and non-fatal for the
        # overall connect() call.
        if not self.node_url:
            logger.warning(
                "DYDX_V4_NODE_URL is not set — running in data-only mode. "
                "Live order placement will be unavailable until it's configured."
            )
            self._connected = True
            return

        try:
            await self._connect_node()
        except Exception as exc:  # noqa: BLE001
            self._node_connected = False
            is_windows_grpc_bug = (
                "unsupported platform" in str(exc).lower()
                or "_InactiveRpcError" in type(exc).__name__
            )
            if is_windows_grpc_bug:
                logger.error(
                    "🪟 Node/gRPC connection failed with a known Windows "
                    "grpcio compatibility issue: %s. Live order placement "
                    "is DISABLED, but market-data reads (candles, "
                    "orderbook, ticker, balance) will continue to work via "
                    "the Indexer REST API. Workarounds: (1) run the bot "
                    "inside WSL2 or Docker Linux containers instead of "
                    "native Windows, (2) pin grpcio to a version with "
                    "confirmed Windows wheel support, or (3) run order "
                    "placement from a Linux/macOS host while this instance "
                    "handles data/analysis only.",
                    exc,
                )
            else:
                logger.error(
                    "Node/gRPC connection failed: %s. Live order placement "
                    "is DISABLED, but market-data reads will continue to "
                    "work via the Indexer REST API.",
                    exc,
                )

        self._connected = True
        logger.info(
            "DydxV4Adapter connected | indexer=%s node=%s address=%s",
            self.indexer_url,
            "CONNECTED" if self._node_connected else "UNAVAILABLE (data-only mode)",
            self._address or "(no wallet — node unavailable)",
        )

    async def _connect_node(self) -> None:
        """Establish the gRPC Node connection and derive the signing wallet.
        Raises on failure; callers decide how to handle that (connect()
        treats it as non-fatal, but callers needing to place orders can
        call this directly to get an eager, explicit failure)."""
        from dydx_v4_client.key_pair import KeyPair
        from dydx_v4_client.network import NodeConfig
        from dydx_v4_client.node.client import NodeClient
        from dydx_v4_client.wallet import Wallet

        async def _do_connect_node() -> None:
            channel = grpc.secure_channel(
                self.node_url, credentials=grpc.ssl_channel_credentials()
            )
            node_config = NodeConfig(
                chain_id=self.chain_id,
                chaintoken_denom="adydx",
                usdc_denom=self.usdc_denom,
                channel=channel,
            )
            self._node = await NodeClient.connect(node_config)

            key_pair = KeyPair.from_mnemonic(self.mnemonic)
            address = Wallet(key=key_pair, account_number=0, sequence=0).address
            self._wallet = await Wallet.from_mnemonic(self._node, self.mnemonic, address)
            self._address = address

        await _with_retries(_do_connect_node, op_name="connect (node/gRPC)")
        self._node_connected = True

    def _ensure_indexer_connected(self) -> None:
        """Guard for read methods that only need the Indexer REST client."""
        if not self._indexer_connected or self._indexer is None:
            raise ConnectionNotEstablishedError(
                "DydxV4Adapter.connect() must be awaited before using this method."
            )

    def _ensure_node_connected(self) -> None:
        """Guard for order-placing methods that need a live gRPC Node
        connection + signing wallet."""
        if not self._node_connected or self._node is None or self._wallet is None:
            raise NodeConnectionError(
                "Node/gRPC connection is not available — cannot place, cancel, "
                "or close orders. This commonly happens on Windows due to a "
                "grpcio platform compatibility issue. Market-data reads "
                "(balance, candles, orderbook, ticker) still work normally. "
                "See the warning logged during connect() for troubleshooting "
                "steps, or run the bot under WSL2/Docker Linux."
            )

    # Backward-compatible alias: existing read methods below call
    # `_ensure_connected()`; keep it as an alias for the indexer-only guard
    # so those call sites don't need to change individually.
    def _ensure_connected(self) -> None:
        self._ensure_indexer_connected()

    @property
    def node_available(self) -> bool:
        """Whether the gRPC Node connection (required for order placement)
        is currently available."""
        return self._node_connected and self._node is not None and self._wallet is not None

    async def close(self) -> None:
        """Close the gRPC channel cleanly (no-op if it was never opened)."""
        if self._node is not None:
            try:
                await self._node.channel.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing gRPC channel: %s", exc)
        self._connected = False
        self._indexer_connected = False
        self._node_connected = False
        logger.info("DydxV4Adapter connection closed.")

    # ------------------------------------------------------------------ #
    # Market metadata (needed to convert price/size -> subticks/quantums)
    # ------------------------------------------------------------------ #

    async def _get_market_meta(self, symbol: str) -> _MarketMeta:
        self._ensure_connected()
        if symbol in self._market_cache:
            return self._market_cache[symbol]

        async def _fetch() -> dict:
            return await self._indexer.markets.get_perpetual_markets(market=symbol)

        response = await _with_retries(_fetch, op_name=f"get_perpetual_markets({symbol})")
        markets = response.get("markets", {})
        market = markets.get(symbol)
        if market is None:
            raise InvalidOrderError(f"Unknown or unlisted dYdX v4 market: {symbol}")

        meta = _MarketMeta(
            ticker=symbol,
            clob_pair_id=int(market["clobPairId"]),
            atomic_resolution=int(market["atomicResolution"]),
            step_base_quantums=int(market["stepBaseQuantums"]),
            quantum_conversion_exponent=int(market["quantumConversionExponent"]),
            subticks_per_tick=int(market["subticksPerTick"]),
            raw=market,
        )
        self._market_cache[symbol] = meta
        return meta

    def _build_market_helper(self, meta: _MarketMeta):
        from dydx_v4_client.node.market import Market

        return Market(
            market={
                "clobPairId": meta.clob_pair_id,
                "atomicResolution": meta.atomic_resolution,
                "stepBaseQuantums": meta.step_base_quantums,
                "quantumConversionExponent": meta.quantum_conversion_exponent,
                "subticksPerTick": meta.subticks_per_tick,
            }
        )

    # ------------------------------------------------------------------ #
    # Read methods
    # ------------------------------------------------------------------ #

    async def get_balance(self) -> float:
        """Fetch free/available USDC equity from the Indexer subaccount."""
        self._ensure_connected()

        async def _fetch() -> dict:
            return await self._indexer.account.get_subaccount(
                address=self._address, subaccount_number=self.subaccount_number
            )

        response = await _with_retries(_fetch, op_name="get_subaccount(balance)")
        subaccount = response.get("subaccount", {})
        free_collateral = subaccount.get("freeCollateral")
        if free_collateral is None:
            raise ExchangeAPIError("Indexer response missing 'freeCollateral' field.")
        return float(free_collateral)

    async def get_account_summary(self) -> dict:
        """
        Return a structured account snapshot matching `PaperExchange`'s
        `get_account_summary()` shape:
            balance_usd, equity_usd, locked_margin_usd, free_margin_usd,
            margin_usage_pct, open_positions, pending_orders
        """
        self._ensure_connected()

        async def _fetch_subaccount() -> dict:
            return await self._indexer.account.get_subaccount(
                address=self._address, subaccount_number=self.subaccount_number
            )

        async def _fetch_orders() -> Any:
            return await self._indexer.account.get_subaccount_orders(
                address=self._address,
                subaccount_number=self.subaccount_number,
                status="OPEN",
            )

        subaccount_resp, orders_resp = await asyncio.gather(
            _with_retries(_fetch_subaccount, op_name="get_subaccount(summary)"),
            _with_retries(_fetch_orders, op_name="get_subaccount_orders"),
        )

        subaccount = subaccount_resp.get("subaccount", {})
        equity = float(subaccount.get("equity", 0.0))
        free_collateral = float(subaccount.get("freeCollateral", 0.0))
        locked_margin = max(equity - free_collateral, 0.0)
        margin_usage_pct = (locked_margin / equity * 100.0) if equity > 0 else 0.0

        open_positions: Dict[str, dict] = {}
        for symbol, pos in (subaccount.get("openPerpetualPositions") or {}).items():
            side = "LONG" if float(pos.get("size", 0.0)) >= 0 else "SHORT"
            open_positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "entry_price": float(pos.get("entryPrice", 0.0)),
                "quantity": abs(float(pos.get("size", 0.0))),
                "leverage": None,  # dYdX v4 cross-margin: not a per-position field
                "stop_loss": None,  # SL is a separate conditional order on-chain
                "take_profit": None,  # TP is a separate conditional order on-chain
                "unrealized_pnl": float(pos.get("unrealizedPnl", 0.0)),
                "realized_pnl": float(pos.get("realizedPnl", 0.0)),
                "margin": None,
            }

        pending_orders: Dict[str, dict] = {}
        for order in (orders_resp or []):
            order_id = order.get("id") or order.get("clientId")
            pending_orders[str(order_id)] = {
                "id": str(order_id),
                "symbol": order.get("ticker", ""),
                "side": order.get("side", ""),
                "order_type": order.get("type", ""),
                "quantity": float(order.get("size", 0.0)),
                "price": float(order.get("price", 0.0)) if order.get("price") else None,
                "status": order.get("status", ""),
                "timestamp": order.get("createdAt", ""),
            }

        return {
            "balance_usd": round(equity, 4),
            "equity_usd": round(equity, 4),
            "locked_margin_usd": round(locked_margin, 4),
            "free_margin_usd": round(free_collateral, 4),
            "margin_usage_pct": round(margin_usage_pct, 2),
            "open_positions": open_positions,
            "pending_orders": pending_orders,
        }

    async def fetch_candles(
        self, symbol: str, resolution: str = "1MIN", limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles directly from the live dYdX v4
        Indexer (`GET /v4/candles/perpetualMarkets/{symbol}`), bypassing
        the SDK client and any intermediate cache — see
        `exchange/indexer_http.py` for why this matters on a tight poll
        loop. Returns a DataFrame shaped
        ['timestamp', 'open', 'high', 'low', 'close', 'volume'].

        `resolution` must be one of dYdX v4's supported values, e.g.
        "1MIN", "5MINS", "15MINS", "30MINS", "1HOUR", "4HOURS", "1DAY"
        (see dydx_v4_client.indexer.candles_resolution.CandlesResolution).
        """
        self._ensure_indexer_connected()

        from exchange.indexer_http import indexer_get

        async def _fetch() -> dict:
            return await indexer_get(
                self.indexer_url,
                f"/v4/candles/perpetualMarkets/{symbol}",
                params={"resolution": resolution, "limit": limit},
            )

        response = await _with_retries(_fetch, op_name=f"get_perpetual_market_candles({symbol})")
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
        Fetch the current order book for `symbol` directly from the
        Indexer (`GET /v4/orderbooks/perpetualMarket/{symbol}`), bypassing
        the SDK client and any intermediate cache.

        Returns the raw Indexer response shape:
            {"bids": [{"price": "...", "size": "..."}, ...],
             "asks": [{"price": "...", "size": "..."}, ...]}
        """
        self._ensure_indexer_connected()

        from exchange.indexer_http import indexer_get

        async def _fetch() -> dict:
            return await indexer_get(
                self.indexer_url, f"/v4/orderbooks/perpetualMarket/{symbol}"
            )

        return await _with_retries(_fetch, op_name=f"get_perpetual_market_orderbook({symbol})")

    async def fetch_ticker_price(self, symbol: str) -> float:
        """
        Fetch the current real-time market price for `symbol` directly
        from the live dYdX v4 Indexer, bypassing the SDK client and any
        intermediate cache (see `exchange/indexer_http.py`).

        Price source priority:
            1. Latest executed trade price
               (`GET /v4/trades/perpetualMarket/{symbol}?limit=1`) — the
               true tick-by-tick market price, updating on every fill.
            2. `oraclePrice`/`indexPrice` from
               `GET /v4/perpetualMarkets?ticker={symbol}` as a fallback.
               Note dYdX's on-chain oracle only updates periodically (not
               per-trade), so polling only that field on a tight loop can
               make the price look artificially constant between oracle
               updates even while the market is actively trading — this
               is why trades are checked first here.

        Note: `place_order`'s internal margin/notional checks intentionally
        use `_get_oracle_price` (oracle price specifically) rather than
        this method, since dYdX's own margin requirement calculations are
        based on the oracle price, not the last trade price.
        """
        self._ensure_indexer_connected()

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
    # Execution
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
    ) -> dict:
        """
        Sign and broadcast an order transaction to the dYdX v4 chain.

        For MARKET/LIMIT entries with `stop_loss`/`take_profit` set, two
        additional conditional orders (STOP_MARKET / TAKE_PROFIT_MARKET,
        reduce-only) are submitted after the entry order, mirroring how
        SL/TP works natively on dYdX v4 (as separate triggered orders
        rather than fields on the entry order itself).

        Raises:
            RuntimeError: if the live-trading kill-switch is not enabled.
            InvalidOrderError: on bad input or unknown market.
            InsufficientFundsError: if free collateral can't cover margin.
            ExchangeAPIError: on unrecoverable gRPC/Indexer failures.
        """
        if not self.live_trading_enabled:
            logger.warning(
                "🛑 LIVE TRADING KILL-SWITCH ACTIVE — refusing to place order "
                "%s %s %s qty=%.6f on %s. Set DYDX_V4_LIVE_TRADING_ENABLED=true "
                "to allow real order submission.",
                side, order_type, symbol, quantity, symbol,
            )
            raise RuntimeError("Live trading kill-switch active")

        self._ensure_node_connected()

        if quantity <= 0:
            raise InvalidOrderError("quantity must be positive")
        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            raise InvalidOrderError(f"invalid side: {side}")
        type_upper = order_type.upper()
        if type_upper not in ("LIMIT", "MARKET"):
            raise InvalidOrderError(f"unsupported order_type for entry: {order_type}")
        if type_upper == "LIMIT" and price is None:
            raise InvalidOrderError("LIMIT orders require a price")

        meta = await self._get_market_meta(symbol)
        market_helper = self._build_market_helper(meta)

        reference_price = price if price is not None else await self._get_oracle_price(symbol)
        notional = quantity * reference_price
        if notional < self.MIN_ORDER_NOTIONAL_USD:
            raise InvalidOrderError(
                f"order notional ${notional:.4f} is below the "
                f"${self.MIN_ORDER_NOTIONAL_USD:.2f} minimum order size"
            )

        free_collateral = await self.get_balance()
        required_margin = notional / max(leverage, 1.0)
        if required_margin > free_collateral:
            raise InsufficientFundsError(
                f"required margin ${required_margin:.4f} exceeds free collateral "
                f"${free_collateral:.4f}"
            )

        from dydx_v4_client import OrderFlags
        from dydx_v4_client.indexer.rest.constants import OrderExecution
        from dydx_v4_client.indexer.rest.constants import OrderType as IndexerOrderType
        from v4_proto.dydxprotocol.clob.order_pb2 import Order as OrderProto

        proto_side = OrderProto.SIDE_BUY if side_upper == "BUY" else OrderProto.SIDE_SELL
        client_id = random.randint(0, 2**32 - 1)

        order_id = market_helper.order_id(
            address=self._address,
            subaccount_number=self.subaccount_number,
            client_id=client_id,
            order_flags=OrderFlags.SHORT_TERM,
        )

        good_til_block = await self._compute_good_til_block()

        entry_indexer_type = (
            IndexerOrderType.MARKET if type_upper == "MARKET" else IndexerOrderType.LIMIT
        )
        entry_order = market_helper.order(
            order_id=order_id,
            order_type=entry_indexer_type,
            side=proto_side,
            size=quantity,
            price=reference_price,
            time_in_force=OrderProto.TIME_IN_FORCE_UNSPECIFIED,
            reduce_only=False,
            good_til_block=good_til_block,
            execution=OrderExecution.IOC if type_upper == "MARKET" else OrderExecution.DEFAULT,
        )

        async def _submit_entry():
            return await self._node.place_order(self._wallet, entry_order)

        tx_result = await _with_retries(_submit_entry, op_name=f"place_order({symbol} entry)")

        logger.info(
            "⚡ ORDER SUBMITTED | %s %s %s qty=%.6f ref_price=%.4f client_id=%d",
            symbol, side_upper, type_upper, quantity, reference_price, client_id,
        )

        result: Dict[str, Any] = {
            "order_id": str(client_id),
            "symbol": symbol,
            "side": side_upper,
            "order_type": type_upper,
            "quantity": quantity,
            "requested_price": reference_price,
            "leverage": leverage,
            "tx_hash": getattr(tx_result, "tx_hash", None) or getattr(tx_result, "txhash", None),
            "stop_loss_order_id": None,
            "take_profit_order_id": None,
        }

        # Opposite side, reduce-only, for both conditional exits.
        exit_side = OrderProto.SIDE_SELL if proto_side == OrderProto.SIDE_BUY else OrderProto.SIDE_BUY

        if stop_loss is not None:
            sl_client_id = random.randint(0, 2**32 - 1)
            sl_order_id = market_helper.order_id(
                address=self._address,
                subaccount_number=self.subaccount_number,
                client_id=sl_client_id,
                order_flags=OrderFlags.CONDITIONAL,
            )
            sl_order = market_helper.order(
                order_id=sl_order_id,
                order_type=IndexerOrderType.STOP_MARKET,
                side=exit_side,
                size=quantity,
                price=stop_loss,
                time_in_force=OrderProto.TIME_IN_FORCE_UNSPECIFIED,
                reduce_only=True,
                good_til_block_time=int(time.time()) + 30 * 24 * 3600,
                execution=OrderExecution.IOC,
                conditional_order_trigger_subticks=market_helper.calculate_subticks(stop_loss),
            )

            async def _submit_sl():
                return await self._node.place_order(self._wallet, sl_order)

            await _with_retries(_submit_sl, op_name=f"place_order({symbol} SL)")
            result["stop_loss_order_id"] = str(sl_client_id)
            logger.info("🛑 STOP-LOSS ORDER SUBMITTED | %s trigger=%.4f", symbol, stop_loss)

        if take_profit is not None:
            tp_client_id = random.randint(0, 2**32 - 1)
            tp_order_id = market_helper.order_id(
                address=self._address,
                subaccount_number=self.subaccount_number,
                client_id=tp_client_id,
                order_flags=OrderFlags.CONDITIONAL,
            )
            tp_order = market_helper.order(
                order_id=tp_order_id,
                order_type=IndexerOrderType.TAKE_PROFIT_MARKET,
                side=exit_side,
                size=quantity,
                price=take_profit,
                time_in_force=OrderProto.TIME_IN_FORCE_UNSPECIFIED,
                reduce_only=True,
                good_til_block_time=int(time.time()) + 30 * 24 * 3600,
                execution=OrderExecution.IOC,
                conditional_order_trigger_subticks=market_helper.calculate_subticks(take_profit),
            )

            async def _submit_tp():
                return await self._node.place_order(self._wallet, tp_order)

            await _with_retries(_submit_tp, op_name=f"place_order({symbol} TP)")
            result["take_profit_order_id"] = str(tp_client_id)
            logger.info("🎯 TAKE-PROFIT ORDER SUBMITTED | %s trigger=%.4f", symbol, take_profit)

        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order by its client id."""
        if not self.live_trading_enabled:
            logger.warning(
                "🛑 LIVE TRADING KILL-SWITCH ACTIVE — refusing to cancel order %s.",
                order_id,
            )
            raise RuntimeError("Live trading kill-switch active")

        self._ensure_node_connected()

        from v4_proto.dydxprotocol.clob.order_pb2 import OrderId as OrderIdProto
        from v4_proto.dydxprotocol.subaccounts.subaccount_pb2 import SubaccountId

        try:
            client_id_int = int(order_id)
        except ValueError as exc:
            raise InvalidOrderError(f"invalid order_id: {order_id}") from exc

        order_id_proto = OrderIdProto(
            subaccount_id=SubaccountId(owner=self._address, number=self.subaccount_number),
            client_id=client_id_int,
            order_flags=0,
            clob_pair_id=0,
        )
        good_til_block = await self._compute_good_til_block()

        async def _submit_cancel():
            return await self._node.cancel_order(
                self._wallet, order_id_proto, good_til_block=good_til_block
            )

        await _with_retries(_submit_cancel, op_name=f"cancel_order({order_id})")
        logger.info("ORDER CANCELLED | id=%s", order_id)
        return True

    async def close_position(self, symbol: str) -> float:
        """
        Market-close an open position for `symbol` by submitting an
        opposite-side, reduce-only MARKET order sized to the full position.
        Returns the position's unrealized PnL at the time of closure
        (actual realized PnL is confirmed asynchronously via the Indexer).
        """
        if not self.live_trading_enabled:
            logger.warning(
                "🛑 LIVE TRADING KILL-SWITCH ACTIVE — refusing to close position %s.",
                symbol,
            )
            raise RuntimeError("Live trading kill-switch active")

        self._ensure_node_connected()

        summary = await self.get_account_summary()
        position = summary["open_positions"].get(symbol)
        if position is None:
            raise InvalidOrderError(f"no open position for {symbol}")

        close_side = "SELL" if position["side"] == "LONG" else "BUY"
        oracle_price = await self._get_oracle_price(symbol)

        await self.place_order(
            symbol=symbol,
            side=close_side,
            order_type="MARKET",
            quantity=position["quantity"],
            price=oracle_price,
        )

        logger.info(
            "✅ POSITION CLOSE SUBMITTED | %s side=%s qty=%.6f ref_price=%.4f",
            symbol, position["side"], position["quantity"], oracle_price,
        )
        return position["unrealized_pnl"]

    # ------------------------------------------------------------------ #
    # Market ticking (no-op for live adapter — the chain streams its own
    # fills/liquidations; this exists only to satisfy BaseExchange so
    # TradingBot can call it unconditionally regardless of backend)
    # ------------------------------------------------------------------ #

    async def on_market_tick(self, symbol: str, current_price: float) -> None:
        """
        No-op for the live adapter: real SL/TP triggers and fills are
        executed on-chain / by the dYdX matching engine, not simulated
        locally. Kept for interface parity with PaperExchange so
        `TradingBot` can call it unconditionally.
        """
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _get_oracle_price(self, symbol: str) -> float:
        """Fetch the current oracle/index price for a market directly from
        the Indexer (HTTP/HTTPS, cache-busted — no Node/gRPC dependency).
        Used internally for margin/notional checks, where dYdX's canonical
        oracle price (not the last trade price) is the correct reference."""
        self._ensure_indexer_connected()

        from exchange.indexer_http import indexer_get

        async def _fetch() -> dict:
            return await indexer_get(
                self.indexer_url, "/v4/perpetualMarkets", params={"ticker": symbol}
            )

        response = await _with_retries(_fetch, op_name=f"get_perpetual_markets(price:{symbol})")
        market = (response.get("markets", {})).get(symbol)
        if market is None:
            raise ExchangeAPIError(f"Could not fetch oracle price for {symbol}: unknown market")

        for price_field in ("oraclePrice", "indexPrice"):
            if market.get(price_field) is not None:
                return float(market[price_field])

        raise ExchangeAPIError(
            f"Could not fetch oracle price for {symbol}: no oraclePrice/indexPrice field"
        )

    async def _compute_good_til_block(self, block_window: int = 20) -> int:
        """
        Short-term orders on dYdX v4 must specify a good-til-block within a
        short window of the current chain height. Fetch the latest height
        and add a small buffer. Requires the Node/gRPC connection.
        """
        self._ensure_node_connected()

        async def _fetch() -> int:
            return await self._node.latest_block_height()

        height = await _with_retries(_fetch, op_name="latest_block_height")
        return height + block_window


# --------------------------------------------------------------------------- #
# Standalone connectivity smoke test (safe: performs read-only calls only,
# and the kill-switch independently blocks any order submission)
# --------------------------------------------------------------------------- #

async def _demo() -> None:
    adapter = DydxV4Adapter()

    if not adapter.mnemonic or not adapter.node_url:
        print(
            "DYDX_V4_MNEMONIC / DYDX_V4_NODE_URL are not configured — "
            "skipping live connectivity demo. Set these environment "
            "variables (and DYDX_V4_INDEXER_URL) to exercise this adapter "
            "against dYdX v4 testnet or mainnet."
        )
        print(f"Live trading kill-switch enabled: {adapter.live_trading_enabled}")
        return

    await adapter.connect()
    try:
        summary = await adapter.get_account_summary()
        print("=== Account Summary ===")
        print(summary)

        candles = await adapter.fetch_candles("ETH-USD", resolution="5MINS", limit=20)
        print("\n=== Recent Candles ===")
        print(candles.tail())
    finally:
        await adapter.close()


if __name__ == "__main__":
    import asyncio as _asyncio

    _asyncio.run(_demo())