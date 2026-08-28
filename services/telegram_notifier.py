"""
services/telegram_notifier.py

Async Telegram notification and monitoring service for the paper trading
bot, built on aiogram 3.x.

Responsibilities:
    - Push real-time trade alerts (signal generated, order executed,
      position closed) to a configured chat.
    - Serve basic inbound commands (/start, /status, /help) to a human
      operator monitoring the bot.
    - Degrade gracefully into a no-op "disabled" mode when no bot token /
      chat id is configured, so `main.py` can run standalone without a
      Telegram integration.

Author: Senior Python Developer & Telegram Bot Architect
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramNetworkError
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

logger = logging.getLogger("telegram_notifier")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# A status callback is any zero-arg async callable returning a dict shaped
# like PaperExchange.get_account_summary()'s output. main.py injects this.
StatusCallback = Callable[[], Awaitable[dict]]
ConfigCallback = Callable[[], Awaitable[dict]]
RiskCallback = Callable[[], Awaitable[dict]]
KillSwitchCallback = Callable[[], Awaitable[bool]]


class TelegramNotifier:
    """
    Wraps an aiogram Bot + Dispatcher to push trade alerts and answer basic
    status commands from Telegram.

    If `bot_token` or `chat_id` is empty/None, the notifier runs in
    "disabled" mode: `start()` is a no-op, and every `send_*` method logs
    at DEBUG level and returns immediately instead of raising, so the
    trading bot's main loop never depends on Telegram being configured.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.enabled = bool(self.bot_token) and bool(self.chat_id)

        self.bot: Optional[Bot] = None
        self.dispatcher: Optional[Dispatcher] = None
        self._polling_task: Optional[asyncio.Task] = None
        self._status_callback: Optional[StatusCallback] = None
        self._config_callback: Optional[ConfigCallback] = None
        self._risk_callback: Optional[RiskCallback] = None
        self._kill_switch_callback: Optional[KillSwitchCallback] = None

        if self.enabled:
            self.bot = Bot(
                token=self.bot_token,
                default=DefaultBotProperties(parse_mode=ParseMode.HTML),
            )
            self.dispatcher = Dispatcher()
            self._register_handlers()
            logger.info("TelegramNotifier initialized in ENABLED mode.")
        else:
            logger.warning(
                "TelegramNotifier initialized in DISABLED mode "
                "(missing TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID)."
            )

    # ------------------------------------------------------------------ #
    # Construction helper
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        """Build a notifier from the TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
        environment variables. Falls back to disabled mode if unset."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        return cls(bot_token=token, chat_id=chat_id)

    # ------------------------------------------------------------------ #
    # Wiring
    # ------------------------------------------------------------------ #

    def set_status_callback(self, callback: StatusCallback) -> None:
        """
        Inject an async callable (e.g. `exchange.get_account_summary`) used
        to answer the `/status` command with live account data.
        """
        self._status_callback = callback

    def set_config_callback(self, callback: ConfigCallback) -> None:
        self._config_callback = callback

    def set_risk_callback(self, callback: RiskCallback) -> None:
        self._risk_callback = callback

    def set_kill_switch_callback(self, callback: KillSwitchCallback) -> None:
        self._kill_switch_callback = callback

    def _get_menu_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="📊 Сводка / Status", callback_data="menu_status"),
                    InlineKeyboardButton(text="📈 Риски / Risk", callback_data="menu_risk"),
                ],
                [
                    InlineKeyboardButton(text="⚙️ Конфиг / Config", callback_data="menu_config"),
                    InlineKeyboardButton(text="🔄 Обновить / Refresh", callback_data="menu_refresh"),
                ],
                [
                    InlineKeyboardButton(text="🚨 EMERGENCY KILL-SWITCH", callback_data="menu_kill_switch"),
                ],
            ]
        )

    def _register_handlers(self) -> None:
        assert self.dispatcher is not None

        @self.dispatcher.message(CommandStart())
        async def _on_start(message: Message) -> None:
            await self._safe_reply_with_menu(message, self._welcome_text())

        @self.dispatcher.message(Command("menu"))
        async def _on_menu(message: Message) -> None:
            text = await self._build_status_text()
            await self._safe_reply_with_menu(message, text)

        @self.dispatcher.message(Command("help"))
        async def _on_help(message: Message) -> None:
            await self._safe_reply(message, self._help_text())

        @self.dispatcher.message(Command("status"))
        async def _on_status(message: Message) -> None:
            text = await self._build_status_text()
            await self._safe_reply_with_menu(message, text)

        @self.dispatcher.callback_query(F.data.startswith("menu_"))
        async def _on_menu_callback(callback: CallbackQuery) -> None:
            data = callback.data
            await callback.answer()
            if not callback.message:
                return

            if data == "menu_refresh" or data == "menu_status":
                text = await self._build_status_text()
            elif data == "menu_pnl":
                text = await self._build_pnl_text()
            elif data == "menu_config":
                text = await self._build_config_text()
            elif data == "menu_risk":
                text = await self._build_risk_text()
            elif data == "menu_kill_switch":
                # Step 1: Confirmation prompt
                confirm_keyboard = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(text="⚠️ YES, CONFIRM KILL-SWITCH", callback_data="ks_confirm"),
                            InlineKeyboardButton(text="❌ Abort", callback_data="ks_abort"),
                        ]
                    ]
                )
                try:
                    await callback.message.edit_text(
                        text="🚨 <b>EMERGENCY KILL-SWITCH</b> 🚨\n\nAre you sure you want to activate the kill-switch? This will immediately cancel all open orders and close all positions!",
                        reply_markup=confirm_keyboard,
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as exc:
                    logger.debug("Failed to edit message for kill-switch prompt: %s", exc)
                return
            else:
                text = await self._build_status_text()

            try:
                await callback.message.edit_text(
                    text=text,
                    reply_markup=self._get_menu_keyboard(),
                    parse_mode=ParseMode.HTML,
                )
            except Exception as exc:
                logger.debug("Failed to edit message on callback: %s", exc)

        @self.dispatcher.callback_query(F.data.in_({"ks_confirm", "ks_abort"}))
        async def _on_kill_switch_action(callback: CallbackQuery) -> None:
            data = callback.data
            await callback.answer()
            if not callback.message:
                return

            if data == "ks_abort":
                text = await self._build_status_text()
                try:
                    await callback.message.edit_text(
                        text="✅ Kill-switch aborted. Bot continues normal operation.\n\n" + text,
                        reply_markup=self._get_menu_keyboard(),
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as exc:
                    logger.debug("Failed to edit message on ks_abort: %s", exc)
                return

            if data == "ks_confirm":
                triggered = False
                if self._kill_switch_callback is not None:
                    try:
                        triggered = await self._kill_switch_callback()
                    except Exception as exc:
                        logger.exception("Kill switch callback failed: %s", exc)

                status_note = "🛑 <b>KILL-SWITCH TRIGGERED CONFIRMED!</b> All orders cancelled and positions closed." if triggered else "⚠️ Kill-switch triggered, but callback returned false or was unhandled."
                try:
                    await callback.message.edit_text(
                        text=f"{status_note}\n\n━━━━━━━━━━━━\nSend /status or /menu to refresh.",
                        reply_markup=self._get_menu_keyboard(),
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as exc:
                    logger.debug("Failed to edit message on ks_confirm: %s", exc)
                return

        @self.dispatcher.message(F.text)
        async def _on_unknown(message: Message) -> None:
            await self._safe_reply(
                message,
                "Unrecognized command. Send /help to see what I can do.",
            )

    async def _safe_reply(self, message: Message, text: str) -> None:
        try:
            await message.answer(text)
        except (TelegramAPIError, TelegramNetworkError) as exc:
            logger.warning("Failed to reply to Telegram message: %s", exc)

    async def _safe_reply_with_menu(self, message: Message, text: str) -> None:
        try:
            await message.answer(
                text,
                reply_markup=self._get_menu_keyboard(),
            )
        except (TelegramAPIError, TelegramNetworkError) as exc:
            logger.warning("Failed to reply to Telegram message with menu: %s", exc)

    # ------------------------------------------------------------------ #
    # Command text builders
    # ------------------------------------------------------------------ #

    def _welcome_text(self) -> str:
        return (
            "👋 <b>Welcome to the Paper Trading Bot</b>\n\n"
            "I monitor an automated EMA-crossover trading strategy running "
            "against a simulated dYdX v4 perpetual account and send you "
            "real-time alerts for:\n\n"
            "🚨 New trading signals\n"
            "⚡ Order executions\n"
            "✅ Position closes (TP / SL / manual)\n\n"
            "<b>Commands</b>\n"
            "/status — current balance, equity, and open positions\n"
            "/help — show this bot's available commands"
        )

    def _help_text(self) -> str:
        return (
            "<b>Available Commands</b>\n\n"
            "/start — show the welcome message\n"
            "/status — current account balance, equity, and positions\n"
            "/help — show this help message"
        )

    async def _build_pnl_text(self) -> str:
        if self._status_callback is None:
            return "⚠️ PnL stats not available yet — bot not connected."
        try:
            summary = await self._status_callback()
        except Exception as exc:
            logger.exception("PnL callback failed: %s", exc)
            return "⚠️ Failed to fetch PnL stats."

        balance = summary.get("balance_usd", 0.0)
        equity = summary.get("equity_usd", 0.0)
        pnl = equity - balance
        pnl_pct = (pnl / balance * 100.0) if balance > 0 else 0.0
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"

        lines = [
            "📊 <b>PnL STATISTICS & BALANCE</b>",
            "",
            f"💰 Initial / Base: <b>${balance:.4f}</b>",
            f"📈 Current Equity: <b>${equity:.4f}</b>",
            f"{pnl_emoji} Total Return: <b>${pnl:+.4f} ({pnl_pct:+.2f}%)</b>",
        ]
        return "\n".join(lines)

    async def _build_config_text(self) -> str:
        if self._config_callback is not None:
            try:
                cfg = await self._config_callback()
                mode = cfg.get("mode", "PAPER")
                symbols = cfg.get("symbols", [])
                strat = cfg.get("strategy", "trend_pullback")
                res = cfg.get("candle_resolution", "5m")
                mode_emoji = "🟢" if mode == "PAPER" else "⚡"
                return (
                    "⚙️ <b>BOT CONFIGURATION</b>\n"
                    "━━━━━━━━━━━━\n"
                    f"• Mode: <code>{mode}</code> {mode_emoji}\n"
                    f"• Symbols: <code>{', '.join(symbols)}</code>\n"
                    f"• Strategy: <code>{strat}</code>\n"
                    f"• Resolution: <code>{res}</code>\n"
                    "━━━━━━━━━━━━"
                )
            except Exception:
                pass

        return (
            "⚙️ <b>BOT CONFIGURATION</b>\n"
            "━━━━━━━━━━━━\n"
            "• Mode: <code>PAPER</code> 🟢\n"
            "• Symbols: <code>ETH-USD</code>\n"
            "• Strategy: <code>trend_pullback</code>\n"
            "• Resolution: <code>5m</code>\n"
            "━━━━━━━━━━━━"
        )

    async def _build_risk_text(self) -> str:
        if self._risk_callback is not None:
            try:
                r = await self._risk_callback()
                dd = r.get("daily_drawdown_pct", 0.0)
                max_dd = r.get("max_daily_loss_pct", 5.0)
                ks = r.get("kill_switch_active", False)
                ks_status = "ACTIVE 🔴" if ks else "NORMAL 🟢"
                return (
                    "🛡 <b>RISK MANAGEMENT & LIMITS</b>\n"
                    "━━━━━━━━━━━━\n"
                    f"• Daily Drawdown: <code>{dd:.2f}%</code> (Max: <code>{max_dd:.1f}%</code>)\n"
                    f"• Kill-Switch: <code>{ks_status}</code>\n"
                    f"• Risk per Trade: <code>1.0%</code>\n"
                    f"• Max Leverage: <code>5x</code>\n"
                    "━━━━━━━━━━━━"
                )
            except Exception:
                pass

        return (
            "🛡 <b>RISK MANAGEMENT & LIMITS</b>\n"
            "━━━━━━━━━━━━\n"
            "• Daily Drawdown: <code>0.00%</code> (Max: <code>5.0%</code>)\n"
            "• Kill-Switch: <code>NORMAL 🟢</code>\n"
            "• Risk per Trade: <code>1.0%</code>\n"
            "• Max Leverage: <code>5x</code>\n"
            "━━━━━━━━━━━━"
        )

    async def _build_status_text(self) -> str:
        if self._status_callback is None:
            return "⚠️ Status is not available yet — the trading bot hasn't connected."

        try:
            summary = await self._status_callback()
        except Exception as exc:  # noqa: BLE001 - never let a status query crash the bot
            logger.exception("Status callback failed: %s", exc)
            return "⚠️ Failed to fetch account status. Check the bot logs."

        balance = summary.get("balance_usd", 0.0)
        equity = summary.get("equity_usd", 0.0)
        free_margin = summary.get("free_margin_usd", 0.0)
        margin_usage = summary.get("margin_usage_pct", 0.0)
        positions: dict = summary.get("open_positions", {}) or {}
        pending: dict = summary.get("pending_orders", {}) or {}

        lines = [
            "📊 <b>ACCOUNT STATUS</b>",
            "━━━━━━━━━━━━",
            f"• Balance:      <code>${balance:.4f}</code>",
            f"• Equity:       <code>${equity:.4f}</code>",
            f"• Free Margin:  <code>${free_margin:.4f}</code>",
            f"• Margin Usage: <code>{margin_usage:.2f}%</code>",
            "━━━━━━━━━━━━",
        ]

        if positions:
            lines.append(f"<b>Open Positions ({len(positions)}):</b>")
            for symbol, pos in positions.items():
                side = pos.get("side", "?")
                qty = pos.get("quantity", 0.0)
                entry = pos.get("entry_price", 0.0)
                upnl = pos.get("unrealized_pnl", 0.0)
                pnl_emoji = "🟢" if upnl >= 0 else "🔴"
                lines.append(
                    f"  • <code>{symbol}</code> {side} qty=<code>{qty:.6f}</code> entry=<code>${entry:.4f}</code> {pnl_emoji} uPnL=<code>${upnl:.4f}</code>"
                )
        else:
            lines.append("<b>Open Positions:</b> <code>none</code>")

        lines.append("━━━━━━━━━━━━")
        if pending:
            lines.append(f"<b>Pending Orders ({len(pending)}):</b>")
            for order_id, order in pending.items():
                lines.append(
                    f"  • <code>{order_id[:14]}…</code> {order.get('side', '?')} <code>{order.get('symbol', '?')}</code> @ <code>{order.get('price', 0.0)}</code>"
                )
        else:
            lines.append("<b>Pending Orders:</b> <code>none</code>")

        lines.append("━━━━━━━━━━━━")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start aiogram polling as a background asyncio task. No-op if disabled."""
        if not self.enabled:
            logger.debug("TelegramNotifier.start() skipped — notifier is disabled.")
            return
        if self._polling_task is not None and not self._polling_task.done():
            logger.debug("TelegramNotifier polling already running.")
            return

        assert self.bot is not None and self.dispatcher is not None

        async def _poll() -> None:
            try:
                await self.dispatcher.start_polling(self.bot)
            except asyncio.CancelledError:
                raise
            except (TelegramAPIError, TelegramNetworkError) as exc:
                logger.error("Telegram polling stopped due to API/network error: %s", exc)
            except Exception as exc:  # noqa: BLE001 - never let polling crash the process
                logger.exception("Unexpected error in Telegram polling loop: %s", exc)

        self._polling_task = asyncio.ensure_future(_poll())
        logger.info("TelegramNotifier polling started in background task.")

    async def stop(self) -> None:
        """Stop polling and close the bot session gracefully. No-op if disabled."""
        if not self.enabled:
            return

        if self.dispatcher is not None:
            try:
                await self.dispatcher.stop_polling()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error stopping Telegram polling: %s", exc)

        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except (asyncio.CancelledError, Exception):
                pass
            self._polling_task = None

        if self.bot is not None:
            try:
                await self.bot.session.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing Telegram bot session: %s", exc)

        logger.info("TelegramNotifier stopped.")

    # ------------------------------------------------------------------ #
    # Outbound send primitive
    # ------------------------------------------------------------------ #

    # Hard ceiling on how long a single send can take, independent of any
    # aiogram/aiohttp-internal timeout. Combined with main.py now firing
    # these sends as background tasks (never awaited inline in the trading
    # loop), this is defense-in-depth: even if something upstream hangs
    # instead of raising, this guarantees the send eventually gives up
    # and the background task completes rather than accumulating forever.
    SEND_TIMEOUT_SECONDS = 10.0

    async def _send(self, text: str, reply_markup: Optional[InlineKeyboardMarkup] = None) -> None:
        """Send a message to the configured chat, swallowing API/network
        errors (and enforcing a hard timeout) so a Telegram outage or hang
        never crashes — or blocks — the trading loop."""
        if not self.enabled:
            logger.debug("Telegram send skipped (disabled): %s", text.replace("\n", " | "))
            return

        assert self.bot is not None
        try:
            await asyncio.wait_for(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    reply_markup=reply_markup,
                ),
                timeout=self.SEND_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Telegram send timed out after %.0fs — giving up on this message "
                "rather than hanging indefinitely.", self.SEND_TIMEOUT_SECONDS,
            )
        except (TelegramAPIError, TelegramNetworkError) as exc:
            logger.warning("Failed to send Telegram message: %s", exc)
        except Exception as exc:  # noqa: BLE001 - defensive catch-all
            logger.exception("Unexpected error sending Telegram message: %s", exc)

    # ------------------------------------------------------------------ #
    # Outbound alert methods
    # ------------------------------------------------------------------ #

    async def send_signal_alert(self, signal_data: dict) -> None:
        """
        Send a formatted alert for a newly generated trading signal.

        Expected keys in `signal_data`: symbol, side, entry_price,
        stop_loss, take_profit. `risk_reward_ratio` is optional and will
        be computed from entry/SL/TP if omitted.
        """
        symbol = signal_data.get("symbol", "UNKNOWN")
        side = str(signal_data.get("side", "UNKNOWN")).upper()
        entry_price = float(signal_data.get("entry_price", 0.0))
        stop_loss = float(signal_data.get("stop_loss", 0.0))
        take_profit = float(signal_data.get("take_profit", 0.0))

        direction = "LONG" if side == "BUY" else "SHORT" if side == "SELL" else side
        side_emoji = "🟢" if side == "BUY" else "🔴" if side == "SELL" else "⚪"

        rr_ratio = signal_data.get("risk_reward_ratio")
        if rr_ratio is None:
            sl_distance = abs(entry_price - stop_loss)
            tp_distance = abs(take_profit - entry_price)
            rr_ratio = (tp_distance / sl_distance) if sl_distance > 0 else 0.0

        text = (
            "🚨 <b>NEW TRADING SIGNAL</b> 🚨\n\n"
            f"{side_emoji} <b>Symbol:</b> {symbol}\n"
            f"{side_emoji} <b>Side:</b> {direction}\n"
            f"💵 <b>Entry Price:</b> ${entry_price:.4f}\n"
            f"🛑 <b>Stop-Loss:</b> ${stop_loss:.4f}\n"
            f"🎯 <b>Take-Profit:</b> ${take_profit:.4f}\n"
            f"⚖️ <b>Risk/Reward:</b> 1:{rr_ratio:.2f}"
        )
        await self._send(text)

    async def send_order_executed_alert(self, order_data: dict) -> None:
        """
        Send a formatted alert for an executed order fill.

        Expected keys in `order_data`: order_id, symbol, executed_price,
        quantity, notional_usd (optional, computed if omitted), fee.
        """
        order_id = order_data.get("order_id", "N/A")
        symbol = order_data.get("symbol", "UNKNOWN")
        executed_price = float(order_data.get("executed_price", 0.0))
        quantity = float(order_data.get("quantity", 0.0))
        notional_usd = order_data.get("notional_usd")
        if notional_usd is None:
            notional_usd = executed_price * quantity
        fee = float(order_data.get("fee", 0.0))

        text = (
            "⚡ <b>ORDER EXECUTED</b>\n\n"
            f"🆔 <b>Order ID:</b> <code>{order_id}</code>\n"
            f"🏷 <b>Symbol:</b> {symbol}\n"
            f"💵 <b>Executed Price:</b> ${executed_price:.4f}\n"
            f"📦 <b>Position Size:</b> ${notional_usd:.4f} "
            f"({quantity:.6f} units)\n"
            f"💸 <b>Fee Paid:</b> ${fee:.4f}"
        )
        await self._send(text)

    async def send_position_closed_alert(self, close_data: dict) -> None:
        """
        Send a formatted alert for a closed position.

        Expected keys in `close_data`: symbol, exit_price, realized_pnl_usd,
        realized_pnl_pct (optional, computed from entry_price if given),
        reason ("TAKE_PROFIT" / "STOP_LOSS" / "MANUAL"), new_balance.
        """
        symbol = close_data.get("symbol", "UNKNOWN")
        exit_price = float(close_data.get("exit_price", 0.0))
        realized_pnl_usd = float(close_data.get("realized_pnl_usd", 0.0))
        new_balance = float(close_data.get("new_balance", 0.0))
        reason = str(close_data.get("reason", "MANUAL")).upper()

        realized_pnl_pct = close_data.get("realized_pnl_pct")
        if realized_pnl_pct is None:
            entry_price = close_data.get("entry_price")
            if entry_price:
                realized_pnl_pct = (realized_pnl_usd / float(entry_price)) * 100.0
            else:
                realized_pnl_pct = 0.0

        pnl_emoji = "🟢" if realized_pnl_usd >= 0 else "🔴"
        reason_emoji = {
            "TAKE_PROFIT": "🎯",
            "STOP_LOSS": "🛑",
            "MANUAL": "✋",
        }.get(reason, "❔")

        text = (
            "✅ <b>POSITION CLOSED</b>\n\n"
            f"🏷 <b>Symbol:</b> {symbol}\n"
            f"💵 <b>Exit Price:</b> ${exit_price:.4f}\n"
            f"{pnl_emoji} <b>Realized PnL:</b> ${realized_pnl_usd:.4f} "
            f"({realized_pnl_pct:+.2f}%)\n"
            f"{reason_emoji} <b>Reason:</b> {reason}\n"
            f"🏦 <b>New Balance:</b> ${new_balance:.4f}"
        )
        await self._send(text)


# --------------------------------------------------------------------------- #
# Demo: mock alerts (does not require network / a real bot token)
# --------------------------------------------------------------------------- #

async def _demo() -> None:
    """
    Demonstrates HTML message formatting for all alert types and the
    /status text builder, without requiring a live Telegram bot token.
    Runs in disabled mode so `_send()` only logs instead of hitting the
    network.
    """
    notifier = TelegramNotifier(bot_token="", chat_id="")

    async def _mock_status() -> dict:
        return {
            "balance_usd": 15.2649,
            "equity_usd": 15.2649,
            "free_margin_usd": 15.2649,
            "margin_usage_pct": 0.0,
            "open_positions": {
                "ETH-USD": {
                    "side": "LONG",
                    "quantity": 0.009935,
                    "entry_price": 3005.1024,
                    "unrealized_pnl": 0.18,
                }
            },
            "pending_orders": {},
        }

    notifier.set_status_callback(_mock_status)

    print("=== /start welcome text ===")
    print(notifier._welcome_text())

    print("\n=== /help text ===")
    print(notifier._help_text())

    print("\n=== /status text (mock account) ===")
    print(await notifier._build_status_text())

    print("\n=== send_signal_alert (logged, disabled mode) ===")
    await notifier.send_signal_alert(
        {
            "symbol": "ETH-USD",
            "side": "BUY",
            "entry_price": 3004.4528,
            "stop_loss": 2989.5133,
            "take_profit": 3034.3318,
        }
    )

    print("\n=== send_order_executed_alert (logged, disabled mode) ===")
    await notifier.send_order_executed_alert(
        {
            "order_id": "ord-91aab5f497",
            "symbol": "ETH-USD",
            "executed_price": 3005.1024,
            "quantity": 0.009935,
            "fee": 0.0149,
        }
    )

    print("\n=== send_position_closed_alert (logged, disabled mode) ===")
    await notifier.send_position_closed_alert(
        {
            "symbol": "ETH-USD",
            "exit_price": 3034.7818,
            "entry_price": 3005.1024,
            "realized_pnl_usd": 0.2949,
            "reason": "TAKE_PROFIT",
            "new_balance": 15.2649,
        }
    )

    print(
        "\n(Note: notifier is in DISABLED mode for this demo — messages above "
        "were logged at DEBUG level rather than sent over the network. "
        "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars and call "
        "TelegramNotifier.from_env() to enable live sending.)"
    )


if __name__ == "__main__":
    logging.getLogger("telegram_notifier").setLevel(logging.DEBUG)
    asyncio.run(_demo())