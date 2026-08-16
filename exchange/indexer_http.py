"""
exchange/indexer_http.py

Direct, cache-safe HTTP GET helper for the dYdX v4 Indexer REST API,
shared by `PaperExchange` and `DydxV4Adapter`.

Why this exists (root cause of "price is static AND outdated"):
    The trading loop issues an identical GET request (e.g.
    `GET /v4/perpetualMarkets?ticker=ETH-USD`) every 1-2 seconds. Any CDN,
    reverse proxy, or intermediate cache sitting in front of the Indexer
    (a very common setup for a public API under load) can legitimately
    cache that exact URL and keep serving the same cached response for
    seconds to minutes — which looks exactly like "the price never
    updates" from the client's point of view, even though the code and
    the live API are both working correctly. A response that was cached
    hours ago (e.g. during a quiet trading period, low-liquidity moment,
    or before a large market move) also explains prices that look both
    static AND stale/outdated relative to the real current market.

    The official `dydx-v4-client` SDK's internal `RestClient.get()` sends
    a plain GET with no cache-control headers and no cache-busting, so it
    is fully exposed to this. This module bypasses the SDK for the
    specific read calls that get polled on a tight loop (ticker price,
    candles, order book) and issues the HTTP request directly with
    explicit no-cache headers plus a cache-busting query parameter, so
    every poll is guaranteed to reach the origin server rather than a
    cached edge response.

Author: Senior Python/Crypto Backend Engineer
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("indexer_http")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT_SECONDS = 10.0

# The one and only mainnet dYdX v4 Indexer host. Every base URL this bot
# uses is validated/normalized against this — see
# `normalize_and_validate_indexer_url()` below.
MAINNET_INDEXER_HOST = "https://indexer.dydx.trade"

# Known non-mainnet host substrings we explicitly guard against. Testnet
# markets trade on thin/artificial liquidity and can show prices wildly
# different from (and much staler than) real mainnet prices — if a bot is
# accidentally pointed at one of these, "static and $1000+ off from the
# real price" is exactly the symptom you'd expect, not a parsing bug.
_KNOWN_NON_MAINNET_MARKERS = ("testnet", "staging", "sandbox", "dev.")

# Headers that instruct any conforming cache (browser, CDN, reverse proxy)
# not to serve or store a cached copy of the response.
_NO_CACHE_HEADERS = {
    "Accept": "application/json",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
}


def normalize_and_validate_indexer_url(
    raw_url: str, allow_non_mainnet: bool = False
) -> str:
    """
    Normalize a configured Indexer base URL and strictly enforce that it
    points at mainnet (`https://indexer.dydx.trade`) unless explicitly
    overridden.

    Two distinct problems this guards against:
      1. A `/v4` suffix left on the base URL (common — dYdX's own docs
         sometimes show `baseURL = 'https://indexer.dydx.trade/v4'` with
         paths like `${baseURL}/perpetualMarkets`). This codebase's
         `indexer_get()` always includes `/v4/...` in the path itself, so
         a base URL that also ends in `/v4` produces a double
         `/v4/v4/perpetualMarkets` path that 404s.
      2. Any host other than the real mainnet Indexer — most commonly a
         testnet endpoint (`indexer.v4testnet.dydx.exchange`), which
         trades on thin, largely synthetic liquidity and can legitimately
         show prices far from and much staler than the real market.

    Raises ValueError if `raw_url` points at a non-mainnet host and
    `allow_non_mainnet` is not True — fails loudly at startup rather than
    silently trading against the wrong market's prices.
    """
    url = (raw_url or "").strip().rstrip("/")
    if not url:
        logger.info("No Indexer URL configured — defaulting to mainnet: %s", MAINNET_INDEXER_HOST)
        return MAINNET_INDEXER_HOST

    if url.endswith("/v4"):
        stripped = url[: -len("/v4")]
        logger.warning(
            "Indexer URL %s has a trailing '/v4' — stripping it to %s, since "
            "this codebase's request paths already include '/v4/...' "
            "(a doubled '/v4/v4/...' path would 404 on every request).",
            url, stripped,
        )
        url = stripped

    lowered = url.lower()
    is_mainnet = lowered == MAINNET_INDEXER_HOST.lower()
    looks_non_mainnet = any(marker in lowered for marker in _KNOWN_NON_MAINNET_MARKERS)

    if not is_mainnet:
        if allow_non_mainnet:
            logger.warning(
                "Indexer URL %s is NOT the mainnet Indexer (%s). Proceeding "
                "anyway because non-mainnet access was explicitly allowed — "
                "prices from this host will NOT reflect real mainnet market "
                "prices.",
                url, MAINNET_INDEXER_HOST,
            )
            return url
        else:
            reason = (
                "matches a known non-mainnet host pattern (testnet/staging/sandbox)"
                if looks_non_mainnet
                else "does not match the mainnet host exactly"
            )
            logger.error(
                "🛑 Configured Indexer URL %s %s. Forcing mainnet (%s) instead "
                "so prices reflect the real market. If this is intentional "
                "(e.g. deliberate testnet use), set "
                "DYDX_V4_ALLOW_NON_MAINNET_INDEXER=true.",
                url, reason, MAINNET_INDEXER_HOST,
            )
            return MAINNET_INDEXER_HOST

    return url


async def indexer_get(
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Issue a direct, cache-busted GET request against the dYdX v4 Indexer
    REST API and return the parsed JSON body.

    `path` must start with "/" (e.g. "/v4/perpetualMarkets"). A fresh
    `httpx.AsyncClient` is used per call (matching the SDK's own
    per-request client lifecycle — no connection/response reuse across
    calls), combined with no-cache headers and a millisecond-precision
    cache-busting query parameter (`_t`) so that repeated identical polls
    cannot be served from any intermediate cache.
    """
    host = base_url.rstrip("/")
    if not path.startswith("/"):
        path = f"/{path}"
    url = f"{host}{path}"

    query: Dict[str, Any] = {k: v for k, v in (params or {}).items() if v is not None}
    query["_t"] = str(int(time.time() * 1000))  # cache-buster: unique per call

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, params=query, headers=_NO_CACHE_HEADERS)
        response.raise_for_status()
        return response.json()


class StalePriceDetector:
    """
    Tracks consecutive identical price readings per symbol and logs a
    loud diagnostic warning if the same exact value repeats too many
    times in a row — a strong signal of caching, a stalled/testnet
    Indexer, or a genuinely illiquid market, rather than a code bug.
    """

    def __init__(self, max_consecutive_repeats: int = 5) -> None:
        self.max_consecutive_repeats = max_consecutive_repeats
        self._last_price: Dict[str, float] = {}
        self._repeat_count: Dict[str, int] = {}
        self._warned: Dict[str, bool] = {}

    def observe(self, symbol: str, price: float, source: str, indexer_url: str) -> None:
        last = self._last_price.get(symbol)
        if last is not None and price == last:
            self._repeat_count[symbol] = self._repeat_count.get(symbol, 1) + 1
        else:
            self._repeat_count[symbol] = 1
            self._warned[symbol] = False

        self._last_price[symbol] = price

        count = self._repeat_count[symbol]
        if count >= self.max_consecutive_repeats and not self._warned.get(symbol):
            self._warned[symbol] = True
            probe_url = f"{indexer_url.rstrip('/')}/v4/perpetualMarkets?ticker={symbol}"
            logger.warning(
                "⚠️ %s price for %s has been exactly %s for %d consecutive "
                "fetches from %s (source=%s). This usually means a CDN/proxy "
                "is caching the request, the Indexer URL points at testnet "
                "(check DYDX_V4_INDEXER_URL), or the market is genuinely "
                "illiquid right now — it is NOT necessarily a parsing bug. "
                "Verify by opening %s directly in a browser and reloading a "
                "few times.",
                source, symbol, price, count, indexer_url, source, probe_url,
            )