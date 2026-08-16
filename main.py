"""
main.py

Autonomous async paper-trading bot orchestrator. Wires together:

    - exchange.paper_exchange.PaperExchange    (simulated dYdX v4 execution)
    - exchange.dydx_v4_adapter.DydxV4Adapter   (live dYdX v4 execution)
    - strategies.trend_ema.TrendEmaStrategy    (EMA crossover + ATR SL/TP)
    - risk.manager.RiskManager                 (equity-based position sizing)
    - services.telegram_notifier.TelegramNotifier (real-time Telegram alerts)

against a synthetic live market data feed, running a continuous async
event loop that ticks the exchange, evaluates the strategy when flat,
sizes/places orders through the risk manager, and pushes Telegram alerts
for signals, fills, and position closes.

Run:
    python3 main.py

Telegram integration is optional: if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
are not set in the environment, TelegramNotifier runs in disabled mode and
the bot operates exactly as before, with alert calls becoming no-ops.

Exchange backend is selected via DYDX_V4_LIVE_TRADING_ENABLED: unset/false
(default) runs against PaperExchange for local simulation; true switches
to DydxV4Adapter for live dYdX v4 execution (which carries its own
independent kill-switch check inside every order-placing call).

Author: Senior Python Async Developer
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
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
from services.telegram_notifier import TelegramNotifier
from strategies.trend_ema import Signal, TrendEmaStrategy
from strategies.trend_pullback import TrendPullbackStrategy

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


# --------------------------------------------------------------------------- #
# Synthetic market data feed
# --------------------------------------------------------------------------- #
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Trading Bot is alive!")

def start_health_check_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# Запускаем фейковый веб-сервер в фоновом потоке
threading.Thread(target=start_health_check_server, daemon=True).start()
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

    def __init__(self, symbol: str = "ETH-USD", initial_balance: float = 15.0) -> None:
        self.symbol = symbol

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

        self.exchange: BaseExchange
        if self.live_trading_enabled:
            self.exchange = DydxV4Adapter()
            logger.info("TradingBot initialized in LIVE dYdX v4 mode")
        else:
            self.exchange = PaperExchange(initial_balance=initial_balance)
            logger.info("TradingBot initialized in PAPER simulation mode")

        self.risk_manager = RiskManager(
            risk_per_trade_pct=1.0,
            max_position_leverage=2.0,
            max_daily_loss_pct=settings.risk_max_daily_loss_pct,
        )

        # Strategy selection: "trend_pullback" (default) trades pullbacks
        # with an EMA200 macro-trend filter instead of raw EMA crossovers
        # — backtesting showed EMA(9/21) crossovers on 1m/5m generate too
        # much noise/fee drag for positive expectancy. "trend_ema" is kept
        # available as the legacy option via STRATEGY_TYPE=trend_ema.
        if settings.strategy_type == "trend_pullback":
            self.strategy = TrendPullbackStrategy(
                ema_trend=settings.strategy_ema_trend,
                ema_pullback=settings.strategy_ema_pullback,
                rsi_period=settings.strategy_rsi_period,
                rsi_oversold=settings.strategy_rsi_oversold,
                rsi_overbought=settings.strategy_rsi_overbought,
                use_rsi_confirmation=settings.strategy_use_rsi_confirmation,
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

        self._running = False
        self._tick_number = 0

        logger.info(
            "TradingBot initialized | symbol=%s initial_balance=$%.2f telegram=%s "
            "exchange=%s candle_resolution=%s",
            symbol, initial_balance, "ENABLED" if self.notifier.enabled else "DISABLED",
            type(self.exchange).__name__, self.candle_resolution,
        )

    # ------------------------------------------------------------------ #
    # Telegram alert helpers (guarded so a Telegram issue never bubbles
    # up into the trading loop)
    # ------------------------------------------------------------------ #

    async def _notify_signal(self, signal: Signal) -> None:
        try:
            await self.notifier.send_signal_alert(signal.to_dict())
        except Exception as exc:  # noqa: BLE001 - never let alerts break trading
            logger.warning("Failed to send signal alert: %s", exc)

    async def _notify_order_executed(
        self,
        order_id: str,
        symbol: str,
        executed_price: float,
        quantity: float,
        fee: float,
    ) -> None:
        try:
            await self.notifier.send_order_executed_alert(
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "executed_price": executed_price,
                    "quantity": quantity,
                    "fee": fee,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to send order executed alert: %s", exc)

    async def _notify_position_closed(
        self,
        symbol: str,
        exit_price: float,
        entry_price: float,
        realized_pnl_usd: float,
        reason: str,
        new_balance: float,
    ) -> None:
        try:
            await self.notifier.send_position_closed_alert(
                {
                    "symbol": symbol,
                    "exit_price": exit_price,
                    "entry_price": entry_price,
                    "realized_pnl_usd": realized_pnl_usd,
                    "reason": reason,
                    "new_balance": new_balance,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to send position closed alert: %s", exc)

    # ------------------------------------------------------------------ #
    # Strategy evaluation + order placement
    # ------------------------------------------------------------------ #

    async def _maybe_open_position(self, df: pd.DataFrame) -> None:
        """Run the strategy when flat and place a sized order on BUY/SELL."""
        try:
            signal: Signal = self.strategy.analyze(self.symbol, df)
        except Exception as exc:  # InsufficientDataError / InvalidDataFrameError
            logger.debug("Strategy analysis skipped: %s", exc)
            return

        if signal.side == "HOLD":
            return

        await self._notify_signal(signal)

        summary = await self.exchange.get_account_summary()
        equity = summary["equity_usd"]
        free_margin = summary["free_margin_usd"]

        plan: PositionPlan = self.risk_manager.calculate_position(
            symbol=self.symbol,
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
                self.symbol, signal.side, plan.rejection_reason,
            )
            return

        order_side = "BUY" if signal.side == "BUY" else "SELL"
        balance_before = summary["balance_usd"]

        try:
            await self.exchange.place_order(
                symbol=self.symbol,
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

        # Order placed successfully (MARKET orders fill synchronously inside
        # place_order). Pull the resulting position + fee delta to report
        # the actual executed price rather than the requested entry price.
        post_summary = await self.exchange.get_account_summary()
        position = post_summary["open_positions"].get(self.symbol)
        if position is not None:
            fee_paid = max(balance_before - post_summary["balance_usd"], 0.0)
            await self._notify_order_executed(
                order_id=f"{self.symbol}-{self._tick_number}",
                symbol=self.symbol,
                executed_price=position["entry_price"],
                quantity=position["quantity"],
                fee=fee_paid,
            )

    # ------------------------------------------------------------------ #
    # Status pulse
    # ------------------------------------------------------------------ #

    async def _status_pulse(self, current_price: float) -> None:
        summary = await self.exchange.get_account_summary()
        positions = summary["open_positions"]
        pending = summary["pending_orders"]

        position_desc = "flat"
        if self.symbol in positions:
            pos = positions[self.symbol]
            position_desc = (
                f"{pos['side']} qty={pos['quantity']:.6f} "
                f"entry={pos['entry_price']:.2f} uPnL=${pos['unrealized_pnl']:.4f}"
            )

        daily_stats = self.risk_manager.get_daily_stats()
        risk_desc = (
            f"daily_dd={daily_stats['daily_drawdown_pct']:.2f}%/"
            f"{daily_stats['max_daily_loss_pct']:.1f}%"
        )
        if daily_stats["kill_switch_active"]:
            risk_desc += " 🛑BLOCKED"

        logger.info(
            "PULSE | price=$%.2f | balance=$%.4f | equity=$%.4f | "
            "position=[%s] | pending_orders=%d | %s",
            current_price, summary["balance_usd"], summary["equity_usd"],
            position_desc, len(pending), risk_desc,
        )

    # ------------------------------------------------------------------ #
    # Main tick
    # ------------------------------------------------------------------ #

    async def _fetch_market_data(self) -> Optional[tuple[float, pd.DataFrame]]:
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
            # strategy its OHLCV window. Both hit the real Indexer.
            current_price, df = await asyncio.gather(
                self.exchange.fetch_ticker_price(self.symbol),
                self.exchange.fetch_candles(symbol=self.symbol, resolution=self.candle_resolution, limit=40),
            )
        except Exception as exc:  # noqa: BLE001 - transient network/API issue
            logger.warning(
                "Failed to fetch real-time market data for %s — skipping this tick: %s",
                self.symbol, exc,
            )
            return None

        if df is None or df.empty:
            logger.warning(
                "fetch_candles returned no data for %s — skipping this tick.",
                self.symbol,
            )
            return None

        return float(current_price), df

    async def _tick_once(self) -> None:
        # Step 1: fetch latest real-time price + OHLCV from the live dYdX
        # v4 Indexer — identical data source in both PAPER and LIVE mode.
        market_data = await self._fetch_market_data()
        if market_data is None:
            return  # skip this tick cleanly; loop keeps running
        current_price, df = market_data

        # Snapshot position + balance state before the tick so we can
        # detect an SL/TP-triggered close performed internally by
        # on_market_tick and report accurate exit/PnL details.
        pre_summary = await self.exchange.get_account_summary()
        pre_position = pre_summary["open_positions"].get(self.symbol)
        balance_before = pre_summary["balance_usd"]

        # Step 2: feed the exchange the new price (limit fills + SL/TP checks
        # in PAPER mode, matched against real-time prices; a harmless no-op
        # in LIVE mode, where fills/SL/TP are executed on-chain by dYdX
        # itself rather than simulated here).
        await self.exchange.on_market_tick(self.symbol, current_price)

        # Step 3: check current positions after the tick
        post_summary = await self.exchange.get_account_summary()
        has_open_position = self.symbol in post_summary["open_positions"]

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
                self.strategy.record_stop_out(self.symbol, side, pd.Timestamp.now(tz="UTC"))
            self.risk_manager.record_realized_pnl(realized_pnl_usd, post_summary["balance_usd"])

            await self._notify_position_closed(
                symbol=self.symbol,
                exit_price=current_price,
                entry_price=entry_price,
                realized_pnl_usd=realized_pnl_usd,
                reason=reason,
                new_balance=post_summary["balance_usd"],
            )

        # Step 4 & 5: evaluate strategy + place sized order only when flat
        if not has_open_position:
            await self._maybe_open_position(df)

        # Step 6: status pulse
        await self._status_pulse(current_price)

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

        # Start the Telegram polling task (no-op if disabled) and wire the
        # /status command up to live account data before entering the loop.
        await self.notifier.start()
        self.notifier.set_status_callback(self.exchange.get_account_summary)

        logger.info(
            "TradingBot starting | symbol=%s poll_interval=%.1fs",
            self.symbol, poll_interval_seconds,
        )

        try:
            while self._running:
                self._tick_number += 1
                try:
                    await self._tick_once()
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
            self._running = False

    async def shutdown(self) -> None:
        """Gracefully stop the loop, close the Telegram session and exchange
        connection, and print final trade metrics."""
        self._running = False

        try:
            await self.notifier.stop()
        except Exception as exc:  # noqa: BLE001 - never block shutdown on Telegram
            logger.warning("Error while stopping TelegramNotifier: %s", exc)

        # Close the exchange connection if the backend exposes one
        # (DydxV4Adapter closes its gRPC channel; PaperExchange has no
        # connection to close and simply won't define this method).
        close_method = getattr(self.exchange, "close", None)
        if callable(close_method):
            try:
                await close_method()
            except Exception as exc:  # noqa: BLE001 - never block shutdown
                logger.warning("Error while closing exchange connection: %s", exc)

        summary = await self.exchange.get_account_summary()

        print("\n" + "=" * 60)
        print("TRADING BOT SHUTDOWN — FINAL ACCOUNT SUMMARY")
        print("=" * 60)
        print(f"Symbol:            {self.symbol}")
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
        print("=" * 60 + "\n")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

async def _main() -> None:
    import signal

    bot = TradingBot(symbol="ETH-USD", initial_balance=15.0)
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
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down gracefully.")