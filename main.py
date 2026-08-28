"""
main.py

Autonomous async multi-ticker paper/live trading bot orchestrator. Wires
together:

    - exchange.paper_exchange.PaperExchange    (simulated dYdX v4 execution)
    - exchange.dydx_v4_adapter.DydxV4Adapter   (live dYdX v4 execution)
    - strategies.trend_ema.TrendEmaStrategy    (EMA crossover + ATR SL/TP)
    - strategies.trend_pullback.TrendPullbackStrategy (EMA200 trend filter
      + EMA20/RSI pullback + ADX + ATR-based TP)
    - risk.manager.RiskManager                 (equity-based position sizing)
    - services.telegram_notifier.TelegramNotifier (real-time Telegram alerts)
    - state.persistence.BotStateStore          (restart-safety persistence)

against the live dYdX v4 Indexer, running a continuous async event loop
that, each tick, evaluates every configured ticker SEQUENTIALLY (not
concurrently) using a single shared exchange adapter instance — this
keeps API usage predictable and respects Indexer rate limits regardless
of how many symbols are configured.

Run:
    python3 main.py

Telegram integration is optional: if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
are not set in the environment, TelegramNotifier runs in disabled mode and
the bot operates exactly as before, with alert calls becoming no-ops.

Exchange backend is selected via DYDX_V4_LIVE_TRADING_ENABLED: unset/false
(default) runs against PaperExchange for local simulation; true switches
to DydxV4Adapter for live dYdX v4 execution (which carries its own
independent kill-switch check inside every order-placing call).

Tickers are configured via TICKERS (comma-separated, e.g.
"BTC-USD,ETH-USD,SOL-USD"), defaulting to a single symbol for backward
compatibility with existing single-asset deployments.

State persistence (STATE_PERSISTENCE_ENABLED, default on): cooldown
timestamps, daily risk-tracking, open conditional-order IDs, and (PAPER
mode only) the full simulated account are persisted to a local JSON file
(STATE_FILE_PATH, default "bot_state.json") after every tick round and on
shutdown, and restored on boot — so a Render container restart doesn't
silently reset cooldowns, forget the day's accumulated PnL past the daily
loss circuit breaker, or wipe the paper account.

Author: Senior Python Async Developer
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# Import config FIRST: this triggers .env loading (via python-dotenv) and
# strict validation of every environment variable the bot depends on,
# before any other module reads os.environ. This is the fix for
# "DYDX_V4_LIVE_TRADING_ENABLED=true in my .env is silently ignored" —
# Python never reads .env files on its own, and previously nothing in
# this codebase loaded one explicitly.
from config import settings

from exchange.base import BaseExchange
from exchange.dydx_v4_adapter import DydxV4Adapter
from exchange.paper_exchange import (
    InsufficientFundsError,
    InvalidOrderError,
    PaperExchange,
)
from risk.manager import PositionPlan, RiskManager
from risk.kill_switch import KillSwitch, KillSwitchTriggered
from services.telegram_notifier import TelegramNotifier
from state.persistence import BotStateStore
from strategies.trend_ema import Signal, StrategyError, TrendEmaStrategy
from strategies.trend_pullback import TrendPullbackStrategy
from strategies.volatility_expansion import VolatilityExpansionStrategy
from strategies.regime_trend import RegimeTrendStrategy

logger = logging.getLogger("trading_bot")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

def _run_health_check_server() -> None:
    port = int(os.environ.get("PORT", 10000))
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"OK")
        def log_message(self, format: str, *args) -> None:
            # Suppress HTTP server request logs to keep stdout clean
            pass

    try:
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        logger.info("Health check HTTP server started on port %d", port)
        server.serve_forever()
    except Exception as exc:
        logger.error("Failed to start health check HTTP server on port %d: %s", port, exc)

class MarketDataFeed:
    """
    Simulates a live price tick stream and maintains a rolling OHLCV
    candle DataFrame for a single symbol.

    Behavior: price drifts down from the seed price for a configurable
    number of ticks (to build a downtrend the strategy can later reverse
    out of), then reverses into an uptrend — guaranteeing a fast/slow EMA
    crossover appears as the bot runs, while still adding small per-tick
    random noise so the feed doesn't look artificially smooth.
    """

    def __init__(
        self,
        symbol: str = "ETH-USD",
        seed_price: float = 3000.0,
        candle_interval_seconds: int = 5,
        history_len: int = 40,
        downtrend_ticks: int = 20,
    ) -> None:
        self.symbol = symbol
        self.candle_interval_seconds = candle_interval_seconds
        self.history_len = history_len
        self.downtrend_ticks = downtrend_ticks

        self._tick_count = 0
        self._price = seed_price
        self._current_candle_open_time: Optional[datetime] = None
        self._current_candle: Optional[dict] = None

        self.candles: pd.DataFrame = self._seed_history(seed_price)

    def _seed_history(self, seed_price: float) -> pd.DataFrame:
        """Pre-populate enough historical candles so the strategy has data
        to work with from the very first live tick."""
        rows: List[dict] = []
        price = seed_price + 40.0
        now = datetime.now(timezone.utc) - timedelta(
            seconds=self.candle_interval_seconds * self.history_len
        )

        for i in range(self.history_len):
            price -= 2.0
            open_ = price + 1.0
            close = price
            high = max(open_, close) + 1.5
            low = min(open_, close) - 1.5
            ts = now + timedelta(seconds=self.candle_interval_seconds * i)
            rows.append(
                {
                    "timestamp": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                }
            )

        self._price = price
        return pd.DataFrame(rows)

    def _next_price(self) -> float:
        """Generate the next simulated tick price."""
        self._tick_count += 1
        noise = random.uniform(-1.5, 1.5)

        if self._tick_count <= self.downtrend_ticks:
            drift = -3.0
        else:
            drift = 6.0

        self._price = max(self._price + drift + noise, 0.01)
        return self._price

    def next_tick(self) -> float:
        """
        Advance the feed by one tick: update the current in-progress candle
        (open a new one if the interval has elapsed) and return the latest
        price.
        """
        price = self._next_price()
        now = datetime.now(timezone.utc)

        if (
            self._current_candle is None
            or self._current_candle_open_time is None
            or (now - self._current_candle_open_time).total_seconds()
            >= self.candle_interval_seconds
        ):
            # Close out the previous in-progress candle into history.
            if self._current_candle is not None:
                self.candles = pd.concat(
                    [self.candles, pd.DataFrame([self._current_candle])],
                    ignore_index=True,
                )
                if len(self.candles) > self.history_len:
                    self.candles = self.candles.iloc[-self.history_len :].reset_index(drop=True)

            self._current_candle_open_time = now
            self._current_candle = {
                "timestamp": now,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
            }
        else:
            self._current_candle["close"] = price
            self._current_candle["high"] = max(self._current_candle["high"], price)
            self._current_candle["low"] = min(self._current_candle["low"], price)
            self._current_candle["volume"] += 100.0

        return price

    def get_ohlcv(self) -> pd.DataFrame:
        """
        Return the rolling OHLCV DataFrame including the current
        (still-forming) candle appended as the latest row, so callers
        always see closed history in rows[:-1] and the live candle in
        row[-1].
        """
        if self._current_candle is None:
            return self.candles.copy()
        return pd.concat(
            [self.candles, pd.DataFrame([self._current_candle])],
            ignore_index=True,
        )


# --------------------------------------------------------------------------- #
# Trading bot orchestrator
# --------------------------------------------------------------------------- #

class TradingBot:
    """
    Autonomous paper-trading bot: polls a market data feed, ticks the
    exchange for fills/SL/TP, evaluates the strategy when flat, sizes
    trades via the risk manager, places orders, and pushes real-time
    Telegram alerts for signals, fills, and position closes.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        initial_balance: float = 15.0,
        inter_symbol_delay_seconds: float = 0.25,
    ) -> None:
        # Ticker list: explicit param wins, otherwise config.settings.tickers
        # (parsed from TICKERS env var, e.g. "BTC-USD,ETH-USD,SOL-USD").
        # Sequential (not concurrent) processing per tick — see _tick_once.
        self.symbols: Tuple[str, ...] = tuple(symbols) if symbols else settings.tickers
        self.inter_symbol_delay_seconds = inter_symbol_delay_seconds

        # Exchange backend selection: live dYdX v4 execution vs local paper
        # simulation, controlled by DYDX_V4_LIVE_TRADING_ENABLED. Sourced
        # from the centralized, validated `config.settings` (loaded once at
        # import time, with .env support and strict boolean parsing) rather
        # than re-reading os.environ here — this is what makes the flag
        # "strictly" activate the live adapter: config.py raises a loud
        # ConfigError at startup if this is true but required dYdX
        # credentials are missing, instead of the bot silently continuing
        # in paper mode. Note that DydxV4Adapter also carries its own
        # independent kill-switch check on every order-placing call, so
        # this flag is not the only guard against accidental live trading.
        self.live_trading_enabled: bool = settings.dydx_v4_live_trading_enabled

        # Candle resolution for both live/paper market data and strategy
        # evaluation. 1-minute EMA crossovers proved too noisy/fee-heavy
        # in backtesting (see scripts/calibrate_strategy.py); default is
        # now 5-minute candles, configurable via CANDLE_RESOLUTION.
        self.candle_resolution: str = settings.candle_resolution

        self.notifier = TelegramNotifier.from_env()

        # Restart-safety persistence (see state/persistence.py). Disabled
        # entirely via STATE_PERSISTENCE_ENABLED=false if not wanted.
        self.state_store: Optional[BotStateStore] = (
            BotStateStore(settings.state_file_path) if settings.state_persistence_enabled else None
        )

        self.kill_switch = KillSwitch(settings, state_store=self.state_store, notifier=self.notifier)
        if self.kill_switch.is_active:
            raise KillSwitchTriggered(
                reason=self.kill_switch.reason or "Kill switch is active at startup.",
                check_name=self.kill_switch.check_name or "STARTUP_BLOCK",
            )

        # A SINGLE shared exchange adapter instance is used across every
        # symbol — this is what makes sequential per-symbol processing
        # actually respect account-wide rate limits (one Indexer
        # connection, one Node/gRPC connection for LIVE, one subaccount
        # whose cross-margined balance/positions naturally cover all
        # symbols in a single get_account_summary() call).
        self.exchange: BaseExchange
        if self.live_trading_enabled:
            self.exchange = DydxV4Adapter(kill_switch=self.kill_switch)
            logger.info("TradingBot initialized in LIVE dYdX v4 mode")
        else:
            self.exchange = PaperExchange(initial_balance=initial_balance)
            logger.info("TradingBot initialized in PAPER simulation mode")

        self.risk_manager = RiskManager(
            risk_per_trade_pct=settings.risk_per_trade_pct,
            max_position_leverage=settings.risk_max_position_leverage,
            max_daily_loss_pct=settings.risk_max_daily_loss_pct,
        )

        # Strategy selection: "trend_pullback" (default) trades pullbacks
        # with an EMA200 macro-trend filter instead of raw EMA crossovers
        # — backtesting showed EMA(9/21) crossovers on 1m/5m generate too
        # much noise/fee drag for positive expectancy. "trend_ema" is kept
        # available as the legacy option via STRATEGY_TYPE=trend_ema.
        # A SINGLE strategy instance is shared across all symbols — its
        # cooldown state is already keyed per-symbol internally
        # (`Dict[symbol][side]`), so this is safe: no state bleed between
        # tickers, and the daily-loss circuit breaker in RiskManager is
        # intentionally portfolio-wide (shared across all symbols), not
        # per-symbol, since it's a single account's daily drawdown limit.
        if settings.strategy_type == "volatility_expansion":
            self.strategy = VolatilityExpansionStrategy(
                n_donchian=settings.strategy_n_donchian,
                n_bb=settings.strategy_n_bb,
                bb_mult=settings.strategy_bb_mult,
                n_percentile_lookback=settings.strategy_n_percentile_lookback,
                compression_percentile_threshold=settings.strategy_compression_percentile_threshold,
                adx_period=settings.strategy_adx_period,
                adx_min_for_entry=settings.strategy_adx_min_for_entry,
                n_vol_ma=settings.strategy_n_vol_ma,
                volume_confirm_mult=settings.strategy_volume_confirm_mult,
                atr_period=settings.strategy_atr_period,
                atr_sl_mult=settings.strategy_atr_sl_mult,
                atr_tp_mult=settings.strategy_atr_tp_mult,
                max_hold_bars=settings.strategy_max_hold_bars,
                cooldown_candles=settings.strategy_cooldown_candles,
                min_atr_pct=settings.strategy_min_atr_pct,
            )
        elif settings.strategy_type == "regime_trend":
            self.strategy = RegimeTrendStrategy(
                ema_fast=20,
                ema_slow=100,
                adx_period=14,
                adx_min=22.0,
                adx_lookback_bars=5,
                atr_period=14,
                atr_sl_mult=1.5,
                atr_tp_mult=3.0,
                max_hold_bars=72,
                cooldown_bars=6,
            )
        elif settings.strategy_type == "trend_pullback":
            self.strategy = TrendPullbackStrategy(
                ema_trend=settings.strategy_ema_trend,
                ema_pullback=settings.strategy_ema_pullback,
                rsi_period=settings.strategy_rsi_period,
                rsi_oversold=settings.strategy_rsi_oversold,
                rsi_overbought=settings.strategy_rsi_overbought,
                use_rsi_confirmation=settings.strategy_use_rsi_confirmation,
                atr_multiplier_sl=settings.strategy_atr_multiplier_sl,
                atr_multiplier_tp=settings.strategy_atr_multiplier_tp,
                use_dynamic_atr_stops=settings.strategy_use_dynamic_atr_stops,
                tp_atr_multiplier=settings.strategy_tp_atr_multiplier,
                adx_period=settings.strategy_adx_period,
                adx_threshold=settings.strategy_adx_threshold,
                use_adx_filter=settings.strategy_use_adx_filter,
                volume_ma_period=settings.strategy_volume_ma_period,
                volume_spike_threshold=settings.strategy_volume_spike_threshold,
                use_volume_confirmation=settings.strategy_use_volume_confirmation,
                cooldown_candles=settings.strategy_cooldown_candles,
                min_atr_pct=settings.strategy_min_atr_pct,
            )
        else:
            self.strategy = TrendEmaStrategy(
                fast_ema=9,
                slow_ema=21,
                cooldown_candles=settings.strategy_cooldown_candles,
                min_atr_pct=settings.strategy_min_atr_pct,
                confirmation_candles=settings.strategy_confirmation_candles,
            )
        # Note: both PaperExchange and DydxV4Adapter now source real-time
        # price data directly from the live dYdX v4 Indexer via
        # fetch_ticker_price()/fetch_candles(); there is no synthetic
        # price feed in the trading loop. MarketDataFeed (below) is kept
        # in this module only as an optional offline/backtesting utility
        # and is not used by TradingBot.
        self.notifier = TelegramNotifier.from_env()

        # Restart-safety persistence (see state/persistence.py). Disabled
        # entirely via STATE_PERSISTENCE_ENABLED=false if not wanted.
        self.state_store: Optional[BotStateStore] = (
            BotStateStore(settings.state_file_path) if settings.state_persistence_enabled else None
        )

        self.kill_switch = KillSwitch(settings, state_store=self.state_store, notifier=self.notifier)
        if self.kill_switch.is_active:
            raise KillSwitchTriggered(
                reason=self.kill_switch.reason or "Kill switch is active at startup.",
                check_name=self.kill_switch.check_name or "STARTUP_BLOCK",
            )

        self._running = False
        self._tick_number = 0
        # symbol -> {"stop_loss_order_id", "take_profit_order_id"} for the
        # currently-open position's conditional orders (LIVE mode only —
        # used to cancel the sibling leg once a position closes, since
        # dYdX v4 does not auto-cancel it; see _cancel_stale_conditional_orders.
        self._open_conditional_orders: Dict[str, dict] = {}
        # Strong references to fire-and-forget background tasks (Telegram
        # alerts) — asyncio only holds a weak reference to a task created
        # via create_task/ensure_future, so without this the task can be
        # garbage-collected mid-flight. See _fire_and_forget().
        self._background_tasks: set = set()

        logger.info(
            "TradingBot initialized | symbols=%s initial_balance=$%.2f telegram=%s "
            "exchange=%s candle_resolution=%s state_persistence=%s",
            list(self.symbols), initial_balance,
            "ENABLED" if self.notifier.enabled else "DISABLED",
            type(self.exchange).__name__, self.candle_resolution,
            "ENABLED" if self.state_store else "DISABLED",
        )

    # ------------------------------------------------------------------ #
    # Telegram alert helpers (guarded so a Telegram issue never bubbles
    # up into the trading loop)
    # ------------------------------------------------------------------ #

    def _fire_and_forget(self, coro, description: str) -> None:
        """
        Schedule `coro` as a background task instead of awaiting it inline.
        This is what actually guarantees a slow/hung Telegram API call can
        NEVER block the main trading loop — a plain `await
        self.notifier.send_*(...)` would stall SL/TP checks and order
        placement for every OTHER symbol in the same tick round until
        Telegram responds (or its internal timeout fires, if any).

        Keeps a strong reference in `self._background_tasks` (asyncio only
        holds a WEAK reference to a task created via create_task/
        ensure_future — without an explicit strong reference held
        elsewhere, the task object can be garbage-collected mid-flight,
        silently cancelling it) and logs any exception the task raises via
        a done-callback, since nothing else will ever await it to surface
        that exception otherwise.
        """
        task = asyncio.ensure_future(coro)
        self._background_tasks.add(task)

        def _on_done(t: "asyncio.Task") -> None:
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.warning("Background task failed (%s): %s", description, exc)

        task.add_done_callback(_on_done)

    async def _drain_background_tasks(self, timeout: float = 5.0) -> None:
        """Best-effort wait for in-flight fire-and-forget tasks (e.g. a
        final position-closed alert) to finish before shutdown, without
        blocking shutdown indefinitely if one is stuck."""
        if not self._background_tasks:
            return
        try:
            done, pending = await asyncio.wait(self._background_tasks, timeout=timeout)
            if pending:
                logger.warning("Cancelling %d stuck background task(s) on shutdown timeout", len(pending))
                for task in pending:
                    task.cancel()
                # Даем секунду на корректное завершение отмененных задач
                await asyncio.gather(*pending, return_exceptions=True)
        except Exception as exc:  # noqa: BLE001 - never block shutdown on this
            logger.warning("Error while draining background tasks: %s", exc)

    def _notify_error_fire_and_forget(self, text: str) -> None:
        """Best-effort raw-text alert for unexpected (non-routine) errors,
        e.g. a genuine bug surfacing from strategy.analyze(). Uses the
        notifier's underlying _send() directly since this isn't one of
        the three structured alert types."""
        send_method = getattr(self.notifier, "_send", None)
        if callable(send_method):
            self._fire_and_forget(send_method(text), description="error alert")

    async def _notify_signal(self, signal: Signal) -> None:
        self._fire_and_forget(
            self.notifier.send_signal_alert(signal.to_dict()),
            description=f"signal alert ({signal.symbol})",
        )

    async def _cancel_stale_conditional_orders(self, symbol: str, triggered_reason: str) -> None:
        """
        Cancel conditional orders after a position closes. For standard SL/TP
        triggers, the opposite leg is cancelled as a stale order. For forced
        exits (e.g. REGIME_INVALIDATION, MAX_HOLD), both legs (SL and TP)
        are cancelled to ensure no orphaned orders are left resting on dYdX v4.
        """
        ids = self._open_conditional_orders.pop(symbol, None)
        if not ids:
            return

        cancel_method = getattr(self.exchange, "cancel_order", None)
        if not callable(cancel_method):
            return

        stale_ids = []
        if triggered_reason == "STOP_LOSS":
            if ids.get("take_profit_order_id"):
                stale_ids.append(ids["take_profit_order_id"])
        elif triggered_reason == "TAKE_PROFIT":
            if ids.get("stop_loss_order_id"):
                stale_ids.append(ids["stop_loss_order_id"])
        else:
            # Forced exit (REGIME_INVALIDATION, MAX_HOLD, etc.) — cancel BOTH legs
            if ids.get("stop_loss_order_id"):
                stale_ids.append(ids["stop_loss_order_id"])
            if ids.get("take_profit_order_id"):
                stale_ids.append(ids["take_profit_order_id"])

        for stale_id in stale_ids:
            try:
                await cancel_method(stale_id)
                logger.info(
                    "🧹 Cancelled stale conditional order %s for %s (position closed via %s)",
                    stale_id, symbol, triggered_reason,
                )
            except Exception as exc:  # noqa: BLE001 - may already be gone
                logger.warning(
                    "Could not cancel conditional order %s for %s (may already be gone): %s",
                    stale_id, symbol, exc,
                )

    async def _notify_order_executed(
        self,
        order_id: str,
        symbol: str,
        executed_price: float,
        quantity: float,
        fee: float,
    ) -> None:
        self._fire_and_forget(
            self.notifier.send_order_executed_alert(
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "executed_price": executed_price,
                    "quantity": quantity,
                    "fee": fee,
                }
            ),
            description=f"order executed alert ({symbol})",
        )

    async def _notify_position_closed(
        self,
        symbol: str,
        exit_price: float,
        entry_price: float,
        realized_pnl_usd: float,
        reason: str,
        new_balance: float,
    ) -> None:
        self._fire_and_forget(
            self.notifier.send_position_closed_alert(
                {
                    "symbol": symbol,
                    "exit_price": exit_price,
                    "entry_price": entry_price,
                    "realized_pnl_usd": realized_pnl_usd,
                    "reason": reason,
                    "new_balance": new_balance,
                }
            ),
            description=f"position closed alert ({symbol})",
        )

    # ------------------------------------------------------------------ #
    # Strategy evaluation + order placement
    # ------------------------------------------------------------------ #

    async def _maybe_open_position(self, symbol: str, df: pd.DataFrame) -> None:
        """Run the strategy when flat and place a sized order on BUY/SELL."""
        try:
            signal: Signal = self.strategy.analyze(symbol, df)
        except StrategyError as exc:
            # Expected, routine conditions (not enough warmed-up candles
            # yet, missing OHLCV columns) — genuinely fine to happen every
            # tick during warmup, so DEBUG is appropriate here.
            logger.debug("Strategy analysis skipped for %s: %s", symbol, exc)
            return
        except Exception as exc:  # noqa: BLE001
            # Anything else (TypeError/KeyError/AttributeError from
            # malformed or unexpectedly-shaped candle data, a real bug in
            # the strategy, etc.) must NEVER be silently swallowed — this
            # is exactly the "silent failure" a production audit needs to
            # catch. Logged at ERROR with full traceback and surfaced to
            # Telegram (best-effort) so an operator actually sees it
            # instead of the bot quietly skipping every tick for a symbol.
            logger.error(
                "UNEXPECTED error during strategy.analyze() for %s — this is NOT a "
                "routine condition, investigate: %s", symbol, exc, exc_info=True,
            )
            self._notify_error_fire_and_forget(
                f"⚠️ Unexpected strategy error for {symbol}: {exc!r}"
            )
            return

        if signal.side == "HOLD":
            return

        await self._notify_signal(signal)

        summary = await self.exchange.get_account_summary()
        equity = summary["equity_usd"]
        free_margin = summary["free_margin_usd"]

        plan: PositionPlan = self.risk_manager.calculate_position(
            symbol=symbol,
            side=signal.side,
            equity_usd=equity,
            free_margin_usd=free_margin,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        if not plan.valid:
            logger.warning(
                "SIGNAL SKIPPED | %s %s reason=%s",
                symbol, signal.side, plan.rejection_reason,
            )
            return

        order_side = "BUY" if signal.side == "BUY" else "SELL"
        balance_before = summary["balance_usd"]

        try:
            order_result = await self.exchange.place_order(
                symbol=symbol,
                side=order_side,
                order_type="MARKET",
                quantity=plan.quantity,
                price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                leverage=plan.leverage,
            )
        except InsufficientFundsError as exc:
            logger.warning("ORDER REJECTED (insufficient funds) | %s", exc)
            return
        except InvalidOrderError as exc:
            logger.warning("ORDER REJECTED (invalid order) | %s", exc)
            return

        # Track the SL/TP conditional order IDs (LIVE mode only —
        # PaperExchange doesn't return them, order_result defaults handle
        # that). Critical for cleanup: dYdX v4 does NOT auto-cancel the
        # sibling leg when one of SL/TP fires — a stale conditional order
        # left resting after a TP/SL close will silently attach to and
        # fire against whatever position is open later, at a stale
        # trigger price unrelated to that later trade. See
        # `_cancel_stale_conditional_orders` for the cleanup half of this.
        if isinstance(order_result, dict):
            self._open_conditional_orders[symbol] = {
                "stop_loss_order_id": order_result.get("stop_loss_order_id"),
                "take_profit_order_id": order_result.get("take_profit_order_id"),
            }

        # Order placed successfully (MARKET orders fill synchronously inside
        # place_order). Pull the resulting position + fee delta to report
        # the actual executed price rather than the requested entry price.
        post_summary = await self.exchange.get_account_summary()
        position = post_summary["open_positions"].get(symbol)
        if position is not None:
            fee_paid = max(balance_before - post_summary["balance_usd"], 0.0)
            await self._notify_order_executed(
                order_id=f"{symbol}-{self._tick_number}",
                symbol=symbol,
                executed_price=position["entry_price"],
                quantity=position["quantity"],
                fee=fee_paid,
            )

    # ------------------------------------------------------------------ #
    # Status pulse
    # ------------------------------------------------------------------ #

    async def _status_pulse(self, tick_prices: Dict[str, float]) -> None:
        """Log one combined pulse line per tick round covering every
        configured symbol, using a single account-wide summary call
        (dYdX v4's cross-margined subaccount naturally returns every
        open position across every symbol in one response)."""
        summary = await self.exchange.get_account_summary()
        positions = summary["open_positions"]
        pending = summary["pending_orders"]

        per_symbol_desc = []
        for symbol in self.symbols:
            price = tick_prices.get(symbol)
            price_str = f"${price:.2f}" if price is not None else "N/A"
            if symbol in positions:
                pos = positions[symbol]
                per_symbol_desc.append(
                    f"{symbol}={price_str}[{pos['side']} qty={pos['quantity']:.6f} "
                    f"uPnL=${pos['unrealized_pnl']:.4f}]"
                )
            else:
                per_symbol_desc.append(f"{symbol}={price_str}[flat]")

        daily_stats = self.risk_manager.get_daily_stats()
        risk_desc = (
            f"daily_dd={daily_stats['daily_drawdown_pct']:.2f}%/"
            f"{daily_stats['max_daily_loss_pct']:.1f}%"
        )
        if daily_stats["kill_switch_active"]:
            risk_desc += " 🛑BLOCKED"

        logger.info(
            "PULSE | balance=$%.4f | equity=$%.4f | %s | pending_orders=%d | %s",
            summary["balance_usd"], summary["equity_usd"],
            " ".join(per_symbol_desc), len(pending), risk_desc,
        )

    # ------------------------------------------------------------------ #
    # Main tick
    # ------------------------------------------------------------------ #

    async def _fetch_market_data(self, symbol: str) -> Optional[tuple[float, pd.DataFrame]]:
        """
        Fetch real-time market data for the current tick. Used identically
        in both PAPER and LIVE mode — `PaperExchange` and `DydxV4Adapter`
        both implement `fetch_ticker_price()`/`fetch_candles()` against the
        same live dYdX v4 Indexer, so price inputs are 100% real-time
        regardless of which backend is executing trades. Only order
        execution (fills, fees, margin, PnL) differs between the two.

        Returns (current_price, ohlcv_df), or None if the fetch fails
        (caller should skip the tick cleanly rather than crash — a
        transient Indexer/network hiccup should never take down the loop).
        """
        try:
            # fetch_ticker_price gives the freshest tick-level price (oracle
            # price, updates continuously); fetch_candles gives the
            # strategy its OHLCV window. Both hit the real Indexer. These
            # two calls run concurrently WITHIN a symbol (harmless, same
            # symbol, same purpose) — it's ACROSS symbols that stays
            # strictly sequential (see _tick_once).
            current_price, df = await asyncio.gather(
                self.exchange.fetch_ticker_price(symbol),
                self.exchange.fetch_candles(symbol=symbol, resolution=self.candle_resolution, limit=40),
            )
        except Exception as exc:  # noqa: BLE001 - transient network/API issue
            logger.warning(
                "Failed to fetch real-time market data for %s — skipping this tick: %s",
                symbol, exc,
            )
            return None

        if df is None or df.empty:
            logger.warning(
                "fetch_candles returned no data for %s — skipping this tick.",
                symbol,
            )
            return None

        return float(current_price), df

    async def _process_symbol_tick(self, symbol: str) -> Optional[float]:
        """
        Run one full tick's worth of processing for a single symbol:
        fetch data, feed the exchange, detect closes (+ risk-control
        bookkeeping + stale conditional-order cleanup), and evaluate for a
        new entry if flat. Returns the current price for the pulse log, or
        None if the market-data fetch failed (that symbol is simply
        skipped for this tick — it does not affect other symbols).
        """
        market_data = await self._fetch_market_data(symbol)
        if market_data is None:
            return None
        current_price, df = market_data

        # Snapshot position + balance state before the tick so we can
        # detect an SL/TP-triggered close performed internally by
        # on_market_tick and report accurate exit/PnL details.
        pre_summary = await self.exchange.get_account_summary()
        pre_position = pre_summary["open_positions"].get(symbol)
        balance_before = pre_summary["balance_usd"]

        # Feed the exchange the new price (limit fills + SL/TP checks in
        # PAPER mode, matched against real-time prices; a harmless no-op
        # in LIVE mode, where fills/SL/TP are executed on-chain by dYdX
        # itself rather than simulated here).
        await self.exchange.on_market_tick(symbol, current_price)

        post_summary = await self.exchange.get_account_summary()
        has_open_position = symbol in post_summary["open_positions"]

        if self.kill_switch is not None:
            day_start_eq = self.risk_manager._daily_starting_equity or post_summary["equity_usd"]
            await self.kill_switch.check_daily_loss(post_summary["equity_usd"], day_start_eq)

        if has_open_position and hasattr(self.strategy, "check_regime_invalidation"):
            pos = post_summary["open_positions"][symbol]
            if self.strategy.check_regime_invalidation(symbol, df, pos["side"]):
                logger.warning("🛑 Regime invalidated for open position on %s! Closing position.", symbol)
                await self.exchange.close_position(symbol)
                await self._cancel_stale_conditional_orders(symbol, triggered_reason="REGIME_INVALIDATION")
                post_summary = await self.exchange.get_account_summary()
                has_open_position = symbol in post_summary["open_positions"]

        # Detect a position that existed before the tick but is gone now —
        # on_market_tick only closes positions via SL/TP triggers, so this
        # unambiguously means one of those fired.
        if pre_position is not None and not has_open_position:
            realized_pnl_usd = post_summary["balance_usd"] - balance_before
            entry_price = pre_position["entry_price"]
            stop_loss = pre_position["stop_loss"]
            side = pre_position["side"]

            if side == "LONG":
                reason = "STOP_LOSS" if current_price <= (stop_loss or float("-inf")) else "TAKE_PROFIT"
            else:
                reason = "STOP_LOSS" if current_price >= (stop_loss or float("inf")) else "TAKE_PROFIT"

            # Feed the risk controls: arm the strategy's post-stop-out
            # cooldown so this direction isn't immediately re-entered, and
            # update the daily circuit breaker with the realized PnL.
            if reason == "STOP_LOSS":
                self.strategy.record_stop_out(symbol, side, pd.Timestamp.now(tz="UTC"))
            self.risk_manager.record_realized_pnl(realized_pnl_usd, post_summary["balance_usd"])

            if self.kill_switch is not None:
                # Build mock trade history item for consecutive losses check
                trade_history = [{"return_pct": (realized_pnl_usd / (entry_price * pre_position["quantity"])) * 100.0}]
                await self.kill_switch.check_consecutive_losses(trade_history)

            # LIVE-mode safety: the OTHER conditional order (whichever of
            # SL/TP did NOT fire) is still resting on dYdX and will NOT be
            # auto-cancelled by the exchange. Left alone, it can later
            # attach to and fire against an unrelated future position.
            await self._cancel_stale_conditional_orders(symbol, triggered_reason=reason)

            await self._notify_position_closed(
                symbol=symbol,
                exit_price=current_price,
                entry_price=entry_price,
                realized_pnl_usd=realized_pnl_usd,
                reason=reason,
                new_balance=post_summary["balance_usd"],
            )

        # Evaluate strategy + place sized order only when flat.
        if not has_open_position:
            if self.kill_switch is not None:
                await self.kill_switch.check_daily_loss(post_summary["equity_usd"], post_summary["balance_usd"])
            await self._maybe_open_position(symbol, df)

        return current_price

    async def _tick_once(self) -> None:
        """
        One full tick round: every configured symbol is processed
        SEQUENTIALLY (never with asyncio.gather across symbols) — this is
        the deliberate design choice that keeps API usage against the
        Indexer predictable and rate-limit-safe regardless of how many
        tickers are configured, at the cost of the tick round taking
        roughly (per-symbol latency x number of symbols) instead of
        max(per-symbol latency). A small pacing delay between symbols adds
        further headroom. Ends with a single combined pulse log covering
        every symbol from one account-wide summary call.
        """
        tick_prices: Dict[str, float] = {}

        for idx, symbol in enumerate(self.symbols):
            try:
                price = await self._process_symbol_tick(symbol)
                if price is not None:
                    tick_prices[symbol] = price
            except InsufficientFundsError as exc:
                logger.warning("Insufficient funds during %s tick: %s", symbol, exc)
            except InvalidOrderError as exc:
                logger.warning("Invalid order during %s tick: %s", symbol, exc)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the others
                logger.exception("Unexpected error processing %s: %s", symbol, exc)

            if idx < len(self.symbols) - 1:
                await asyncio.sleep(self.inter_symbol_delay_seconds)

        await self._status_pulse(tick_prices)

        if self.state_store is not None:
            await self._save_state()

    # ------------------------------------------------------------------ #
    # Event loop
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # State persistence (restart safety)
    # ------------------------------------------------------------------ #

    def _build_state_dict(self) -> dict:
        """
        Assemble the full persisted-state dict: cooldowns, daily risk
        tracking, open conditional-order IDs, and (PAPER mode only) the
        full simulated account. LIVE mode's positions/balance are NOT
        persisted here — dYdX itself is already the source of truth for
        those, and re-deriving them locally on restart would risk
        drifting out of sync with reality.
        """
        state = {
            "symbols": list(self.symbols),
            "cooldowns": self.strategy.get_cooldown_state(),
            "daily_risk": self.risk_manager.get_state(),
            "open_conditional_orders": dict(self._open_conditional_orders),
        }
        if isinstance(self.exchange, PaperExchange):
            state["paper_account"] = self.exchange.export_state()
        return state

    async def _save_state(self) -> None:
        if self.state_store is None:
            return
        try:
            self.state_store.save(self._build_state_dict())
        except Exception as exc:  # noqa: BLE001 - persistence must never crash the loop
            logger.warning("Failed to persist bot state (continuing without it): %s", exc)

    def _load_and_apply_state(self) -> None:
        """
        Load persisted state (if any) and apply it to the strategy,
        risk manager, and (PAPER mode) exchange BEFORE the trading loop
        starts. Called once from `run()`, after construction but before
        the first tick. A missing/corrupted/schema-mismatched state file
        is handled entirely inside `BotStateStore.load()` — this method
        never needs to worry about that case beyond checking for None.
        """
        if self.state_store is None:
            return

        state = self.state_store.load()
        if state is None:
            return

        saved_symbols = state.get("symbols")
        if saved_symbols and set(saved_symbols) != set(self.symbols):
            logger.warning(
                "Persisted state was saved for symbols=%s but this run is "
                "configured for symbols=%s — restoring only the overlapping "
                "cooldown entries; daily risk state is portfolio-wide and "
                "restores regardless.",
                saved_symbols, list(self.symbols),
            )

        if "cooldowns" in state:
            self.strategy.restore_cooldown_state(state["cooldowns"])

        if "daily_risk" in state:
            self.risk_manager.restore_state(state["daily_risk"])

        self._open_conditional_orders = dict(state.get("open_conditional_orders", {}))

        if isinstance(self.exchange, PaperExchange) and "paper_account" in state:
            self.exchange.import_state(state["paper_account"])

    # ------------------------------------------------------------------ #
    # Event loop
    # ------------------------------------------------------------------ #

    async def run(self, poll_interval_seconds: float = 2.0) -> None:
        """Main autonomous event loop. Runs until cancelled or interrupted."""
        self._running = True

        # Both backends now require an explicit connect() before ticking:
        # DydxV4Adapter connects the Indexer (REST) + Node (gRPC/signing);
        # PaperExchange connects only the Indexer (REST) for real-time
        # market data, since it never signs/broadcasts anything. Calling
        # this uniformly (rather than isinstance-branching) keeps
        # TradingBot exchange-agnostic.
        connect_method = getattr(self.exchange, "connect", None)
        if callable(connect_method):
            logger.info(
                "Connecting to dYdX v4 (%s)...",
                "Indexer + Node" if isinstance(self.exchange, DydxV4Adapter) else "Indexer, real-time data only",
            )
            await connect_method()

        # Restore persisted state (cooldowns, daily risk, paper account)
        # AFTER connect() (so PaperExchange's import_state overwrites the
        # fresh account connect() would otherwise leave untouched — connect()
        # only sets up market-data access, it doesn't touch account state)
        # but BEFORE the first tick.
        self._load_and_apply_state()

        # Start the Telegram polling task (no-op if disabled) and wire the
        # /status command up to live account data before entering the loop.
        await self.notifier.start()
        self.notifier.set_status_callback(self.exchange.get_account_summary)
        
        async def _get_config_summary() -> dict:
            return {
                "mode": "LIVE" if self.live_trading_enabled else "PAPER",
                "symbols": list(self.symbols),
                "strategy": settings.strategy_type,
                "candle_resolution": self.candle_resolution,
            }
        self.notifier.set_config_callback(_get_config_summary)

        async def _get_risk_summary() -> dict:
            return self.risk_manager.get_daily_stats()
        self.notifier.set_risk_callback(_get_risk_summary)

        async def _telegram_kill_switch_callback() -> bool:
            logger.critical("🚨 EMERGENCY KILL-SWITCH ACTIVATED VIA TELEGRAM INLINE BUTTON!")
            if self.kill_switch is not None:
                self.kill_switch.trigger("TELEGRAM_CALLBACK_MANUAL", check_name="TELEGRAM_BUTTON")
            self._running = False
            return True
        self.notifier.set_kill_switch_callback(_telegram_kill_switch_callback)

        logger.info(
            "TradingBot starting | symbols=%s poll_interval=%.1fs",
            list(self.symbols), poll_interval_seconds,
        )

        try:
            try:
                while self._running:
                    self._tick_number += 1
                    try:
                        await self._tick_once()
                    except KillSwitchTriggered:
                        logger.critical("🛑 Kill switch triggered during tick loop! Shutting down.")
                        self._running = False
                        raise
                    except InsufficientFundsError as exc:
                        logger.warning("Insufficient funds during tick: %s", exc)
                    except InvalidOrderError as exc:
                        logger.warning("Invalid order during tick: %s", exc)
                    except Exception as exc:  # noqa: BLE001 - keep the loop alive
                        logger.exception("Unexpected error during tick: %s", exc)

                    await asyncio.sleep(poll_interval_seconds)
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled.")
                raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully stop the loop, persist final state, close the
        Telegram session and exchange connection, and print final trade
        metrics."""
        self._running = False

        # Экстренный клиринг при срабатывании KillSwitch
        if self.kill_switch is not None and getattr(self.kill_switch, "is_active", False):
            logger.critical("🚨 EMERGENCY KILL-SWITCH CLEARING: Cancelling orders and closing open positions...")
            
            # 1. Отменяем все ордера / условные ордера (отдельный try/except блок)
            try:
                cancel_all = getattr(self.exchange, "cancel_all_open_orders", None)
                if callable(cancel_all):
                    await cancel_all()
                else:
                    for sym, ids in list(self._open_conditional_orders.items()):
                        for order_key in ("stop_loss_order_id", "take_profit_order_id"):
                            oid = ids.get(order_key)
                            if oid:
                                cancel_method = getattr(self.exchange, "cancel_order", None)
                                if callable(cancel_method):
                                    try:
                                        await cancel_method(oid)
                                    except Exception:
                                        pass
                    self._open_conditional_orders.clear()
            except Exception as exc:
                logger.exception("Failed to cancel orders during emergency kill-switch clearing: %s", exc)

            # 2. Закрываем все открытые позиции (отдельный try/except блок)
            try:
                summary = await self.exchange.get_account_summary()
                open_positions = summary.get("open_positions", {})
                for sym in list(open_positions.keys()):
                    logger.warning("🚨 EMERGENCY CLOSE POSITION for %s", sym)
                    close_pos_method = getattr(self.exchange, "close_position", None)
                    if callable(close_pos_method):
                        await close_pos_method(sym)
            except Exception as exc:
                logger.exception("Failed to close positions during emergency kill-switch clearing: %s", exc)

        # Give any in-flight fire-and-forget Telegram alerts (e.g. a final
        # position-closed notification) a bounded chance to finish before
        # tearing down the notifier — best-effort, never blocks shutdown
        # indefinitely.
        await self._drain_background_tasks(timeout=5.0)

        await self._save_state()

        try:
            await self.notifier.stop()
        except Exception as exc:  # noqa: BLE001 - never block shutdown on Telegram
            logger.warning("Error while stopping TelegramNotifier: %s", exc)

        # Close the exchange connection if the backend exposes one
        # (DydxV4Adapter closes its gRPC channel; PaperExchange has no
        # connection to close and simply won't define this method).
        close_method = getattr(self.exchange, "close", None)
        
        # Получаем сводку ДО закрытия соединения close_method()
        summary = await self.exchange.get_account_summary()

        if callable(close_method):
            try:
                await close_method()
            except Exception as exc:  # noqa: BLE001 - never block shutdown
                logger.warning("Error while closing exchange connection: %s", exc)

        print("\n" + "=" * 60)
        print("TRADING BOT SHUTDOWN — FINAL ACCOUNT SUMMARY")
        print("=" * 60)
        print(f"Symbols:           {list(self.symbols)}")
        print(f"Exchange:          {type(self.exchange).__name__}")
        print(f"Ticks processed:   {self._tick_number}")
        print(f"Final Balance:     ${summary['balance_usd']:.4f}")
        print(f"Final Equity:      ${summary['equity_usd']:.4f}")
        print(f"Locked Margin:     ${summary['locked_margin_usd']:.4f}")
        print(f"Free Margin:       ${summary['free_margin_usd']:.4f}")
        print(f"Margin Usage:      {summary['margin_usage_pct']:.2f}%")
        print(f"Open Positions:    {list(summary['open_positions'].keys()) or 'none'}")
        for sym, pos in summary["open_positions"].items():
            print(
                f"  - {sym}: {pos['side']} qty={pos['quantity']:.6f} "
                f"entry={pos['entry_price']:.2f} uPnL=${pos['unrealized_pnl']:.4f}"
            )
        print(f"Pending Orders:    {list(summary['pending_orders'].keys()) or 'none'}")
        if self.state_store is not None:
            print(f"State persisted to: {self.state_store.file_path}")
        print("=" * 60 + "\n")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

async def _main() -> None:
    import signal

    bot = TradingBot(initial_balance=settings.paper_balance)
    run_task = asyncio.ensure_future(bot.run(poll_interval_seconds=2.0))

    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()

    def _request_stop(sig_name: str) -> None:
        logger.info("%s received — shutting down gracefully.", sig_name)
        stop_requested.set()

    # Prefer native asyncio signal handlers (reliable delivery inside the
    # running loop); fall back to default KeyboardInterrupt handling on
    # platforms that don't support add_signal_handler (e.g. Windows).
    signal_handlers_installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, _request_stop, "SIGINT")
        loop.add_signal_handler(signal.SIGTERM, _request_stop, "SIGTERM")
        signal_handlers_installed = True
    except (NotImplementedError, RuntimeError):
        pass

    stop_waiter = asyncio.ensure_future(stop_requested.wait())

    try:
        done, pending = await asyncio.wait(
            {run_task, stop_waiter}, return_when=asyncio.FIRST_COMPLETED
        )
    except KeyboardInterrupt:
        stop_requested.set()
        done, pending = set(), {run_task, stop_waiter}

    if stop_waiter in pending:
        stop_waiter.cancel()

    if not run_task.done():
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass

    if signal_handlers_installed:
        loop.remove_signal_handler(signal.SIGINT)
        loop.remove_signal_handler(signal.SIGTERM)

    await bot.shutdown()


if __name__ == "__main__":
    # Start the background HTTP server daemon thread for Render web service health check port binding
    _http_thread = threading.Thread(target=_run_health_check_server, daemon=True)
    _http_thread.start()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down gracefully.")