"""
config.py

Centralized, validated configuration for the trading bot. Loads `.env`
explicitly via python-dotenv (Python does NOT read `.env` files on its
own — this is the #1 cause of "my env vars are set but the bot ignores
them" bugs, especially on Windows where `.env` is never sourced into the
process environment the way it might be on a Unix shell with `export`).

All other modules should import the shared `settings` singleton from here
instead of calling `os.environ.get(...)` directly, so there is exactly
one place that parses and validates configuration.

Author: Senior Python Async Developer
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("config")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class ConfigError(Exception):
    """Raised for invalid or contradictory configuration that should stop
    the bot from starting rather than silently degrading behavior."""


# --------------------------------------------------------------------------- #
# .env loading
# --------------------------------------------------------------------------- #
#
# find_dotenv() walks upward from the current working directory looking
# for a `.env` file, so this works whether the bot is launched as
# `python main.py` from the project root or via an IDE/Windows shortcut
# with a different working directory. override=False means real OS-level
# environment variables (e.g. set in Docker, systemd, or a CI runner)
# always win over `.env` — `.env` only fills in what's not already set.

def _load_dotenv() -> Optional[str]:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        logger.warning(
            "python-dotenv is not installed — .env files will NOT be loaded "
            "automatically. Environment variables must be set directly in "
            "the OS/shell/container. Install with: pip install python-dotenv"
        )
        return None

    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        logger.info(
            "No .env file found (searched from %s upward). Relying on "
            "process environment variables only.",
            Path.cwd(),
        )
        return None

    loaded = load_dotenv(dotenv_path=dotenv_path, override=False)
    if loaded:
        logger.info("Loaded environment variables from %s", dotenv_path)
    else:
        logger.warning(
            "Found .env at %s but python-dotenv reported nothing was "
            "loaded — check the file isn't empty or malformed.",
            dotenv_path,
        )
    return dotenv_path


_DOTENV_PATH = _load_dotenv()


# --------------------------------------------------------------------------- #
# Strict boolean parsing
# --------------------------------------------------------------------------- #
#
# The historical bug this fixes: `bool("false")` in Python is `True`
# (any non-empty string is truthy), and `os.environ.get(...)` returns a
# *string*. Code that ever did `bool(os.environ.get("SOME_FLAG"))` or
# compared against the wrong case would silently misbehave. This helper
# is the single source of truth for turning an env var string into a
# real boolean, with an explicit, case-insensitive allow-list.

_TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}
_FALSE_VALUES = {"0", "false", "no", "off", "n", "f", ""}


def parse_bool_env(name: str, default: bool = False) -> bool:
    """
    Strictly parse an environment variable as a boolean.

    Raises ConfigError if the variable is set to a value that isn't a
    recognized true/false token, rather than silently defaulting — a
    typo like `DYDX_V4_LIVE_TRADING_ENABLED=tru` should fail loudly, not
    quietly disable live trading.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    raise ConfigError(
        f"Environment variable {name}={raw!r} is not a recognized boolean "
        f"value. Use one of: {sorted(_TRUE_VALUES | _FALSE_VALUES - {''})} "
        f"(empty string is treated as false)."
    )


def _get_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}={raw!r} is not a valid integer.") from exc


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}={raw!r} is not a valid float.") from exc


# dYdX v4's real Indexer API uses these exact (inconsistently pluralized —
# "1MIN" but "5MINS") resolution strings. This maps common human-friendly
# aliases to the canonical value the API actually accepts, so
# CANDLE_RESOLUTION=5m / 5min / 5MIN / 5mins / 5MINS all resolve correctly
# instead of silently 404ing against the real endpoint.
_CANDLE_RESOLUTION_ALIASES: Dict[str, str] = {
    "1M": "1MIN", "1MIN": "1MIN", "1MINS": "1MIN", "1MINUTE": "1MIN",
    "5M": "5MINS", "5MIN": "5MINS", "5MINS": "5MINS", "5MINUTES": "5MINS",
    "15M": "15MINS", "15MIN": "15MINS", "15MINS": "15MINS", "15MINUTES": "15MINS",
    "30M": "30MINS", "30MIN": "30MINS", "30MINS": "30MINS", "30MINUTES": "30MINS",
    "1H": "1HOUR", "1HOUR": "1HOUR", "1HOURS": "1HOUR",
    "4H": "4HOURS", "4HOUR": "4HOURS", "4HOURS": "4HOURS",
    "1D": "1DAY", "1DAY": "1DAY", "1DAYS": "1DAY",
}


def normalize_candle_resolution(raw: str) -> str:
    """Normalize a CANDLE_RESOLUTION value (accepting common aliases like
    '5m'/'5min') to the exact string dYdX v4's Indexer API expects.
    Raises ConfigError on anything unrecognized."""
    key = raw.strip().upper()
    canonical = _CANDLE_RESOLUTION_ALIASES.get(key)
    if canonical is None:
        raise ConfigError(
            f"CANDLE_RESOLUTION={raw!r} is not a recognized resolution. "
            f"Supported: 1MIN, 5MINS, 15MINS, 30MINS, 1HOUR, 4HOURS, 1DAY "
            f"(aliases like '5m'/'5min' are also accepted)."
        )
    return canonical


def _parse_tickers(raw: str) -> Tuple[str, ...]:
    """
    Parse a comma-separated ticker list (e.g. "BTC-USD, eth-usd,SOL-USD")
    into a clean, uppercased, deduplicated, order-preserving tuple.
    Raises ConfigError if the result is empty — the bot needs at least
    one symbol to trade.
    """
    seen = set()
    tickers = []
    for raw_symbol in raw.split(","):
        symbol = raw_symbol.strip().upper()
        if not symbol:
            continue
        if symbol not in seen:
            seen.add(symbol)
            tickers.append(symbol)

    if not tickers:
        raise ConfigError(
            f"TICKERS={raw!r} did not produce any valid symbols. Expected a "
            f"comma-separated list like 'BTC-USD,ETH-USD,SOL-USD'."
        )
    return tuple(tickers)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Settings:
    """
    Immutable, validated snapshot of the bot's configuration. Constructed
    once at import time from environment variables (after `.env` has been
    loaded above) via `Settings.load()`.
    """

    # --- Trading mode -------------------------------------------------
    dydx_v4_live_trading_enabled: bool
    paper_balance: float

    # --- dYdX v4 credentials / endpoints --------------------------------
    dydx_v4_mnemonic: str
    dydx_v4_private_key: str
    dydx_v4_node_url: str
    dydx_v4_indexer_url: str
    dydx_v4_chain_id: str
    dydx_v4_usdc_denom: str
    dydx_v4_subaccount_number: int

    # --- Telegram ------------------------------------------------------
    telegram_bot_token: str
    telegram_chat_id: str

    # --- Strategy selection + risk controls -------------------------------
    # See strategies/trend_ema.py, strategies/trend_pullback.py, and
    # risk/manager.py for full semantics.
    strategy_type: str  # "volatility_expansion", "trend_pullback", or "trend_ema"
    strategy_cooldown_candles: int
    strategy_min_atr_pct: float
    strategy_confirmation_candles: int  # trend_ema only
    strategy_ema_trend: int             # trend_pullback only
    strategy_ema_pullback: int          # trend_pullback only
    strategy_rsi_period: int            # trend_pullback only
    strategy_rsi_oversold: float        # trend_pullback only
    strategy_rsi_overbought: float      # trend_pullback only
    strategy_use_rsi_confirmation: bool # trend_pullback only
    strategy_adx_period: int            # trend_pullback / volatility_expansion
    strategy_adx_threshold: float       # trend_pullback only
    strategy_use_adx_filter: bool       # trend_pullback only
    strategy_tp_atr_multiplier: Optional[float]  # trend_pullback only; None = use risk_reward_ratio
    # volatility_expansion specific parameters
    strategy_n_donchian: int
    strategy_n_bb: int
    strategy_bb_mult: float
    strategy_n_percentile_lookback: int
    strategy_compression_percentile_threshold: float
    strategy_adx_min_for_entry: float
    strategy_n_vol_ma: int
    strategy_volume_confirm_mult: float
    strategy_atr_period: int
    strategy_atr_sl_mult: float
    strategy_atr_tp_mult: float
    strategy_max_hold_bars: int
    # trend_pullback only: dynamic ATR-based SL/TP (separate multipliers
    # instead of one shared atr_multiplier + risk_reward_ratio) and volume
    # spike confirmation (require volume >= vol_ma * threshold on the
    # pullback-confirmation candle, to filter out fake pullbacks that
    # happen on thin/low-conviction volume).
    strategy_atr_multiplier_sl: float
    strategy_atr_multiplier_tp: float
    strategy_use_dynamic_atr_stops: bool
    strategy_volume_ma_period: int
    strategy_volume_spike_threshold: float
    strategy_use_volume_confirmation: bool
    risk_max_daily_loss_pct: float
    risk_max_position_leverage: float
    risk_per_trade_pct: float

    # --- Kill Switch --------------------------------------------------------
    kill_max_daily_loss_pct: float
    kill_max_consecutive_losses: int
    kill_max_position_notional_pct: float
    kill_max_slippage_pct: float
    kill_max_orders_per_hour: int
    kill_heartbeat_timeout_sec: float

    # --- Market data -------------------------------------------------------
    # Candle resolution used for both live trading and calibration. Must be
    # one of dYdX v4's supported values: "1MIN", "5MINS", "15MINS",
    # "30MINS", "1HOUR", "4HOURS", "1DAY". Backtesting showed 1-minute EMA
    # crossovers generate too much noise/fee drag for positive expectancy,
    # so the default is now 5-minute candles.
    candle_resolution: str

    # --- Tickers -----------------------------------------------------------
    # Symbols the bot trades, evaluated sequentially each tick (not
    # concurrently) using a single shared exchange adapter instance, to
    # keep API usage predictable and respect rate limits regardless of
    # how many symbols are configured.
    tickers: Tuple[str, ...]

    # --- State persistence (restart safety) ---------------------------------
    state_file_path: str
    state_persistence_enabled: bool

    # --- Misc ------------------------------------------------------------
    log_level: str
    dotenv_path: Optional[str] = field(default=None, repr=False)

    @staticmethod
    def load() -> "Settings":
        # DYDX_V4_LIVE_TRADING_ENABLED remains the primary flag (matches
        # the rest of the dYdX-specific naming in this file), but
        # PAPER_TRADING is also accepted as a more intuitive, explicitly
        # requested alias — PAPER_TRADING=true forces paper mode
        # regardless of DYDX_V4_LIVE_TRADING_ENABLED; PAPER_TRADING=false
        # is equivalent to DYDX_V4_LIVE_TRADING_ENABLED=true. If PAPER_TRADING
        # is not set at all, DYDX_V4_LIVE_TRADING_ENABLED alone decides.
        live_trading_enabled = parse_bool_env("DYDX_V4_LIVE_TRADING_ENABLED", default=False)
        if os.environ.get("PAPER_TRADING", "").strip():
            paper_trading = parse_bool_env("PAPER_TRADING", default=True)
            live_trading_enabled = not paper_trading
        mnemonic = _get_str("DYDX_V4_MNEMONIC")
        private_key = _get_str("DYDX_V4_PRIVATE_KEY")

        # Normalize/validate the Indexer URL once, centrally, so every
        # consumer of `settings.dydx_v4_indexer_url` gets an already-strict
        # mainnet URL (strips a stray trailing '/v4', forces mainnet unless
        # DYDX_V4_ALLOW_NON_MAINNET_INDEXER=true is set). See
        # exchange/indexer_http.py for the full rationale.
        try:
            from exchange.indexer_http import normalize_and_validate_indexer_url
            allow_non_mainnet = parse_bool_env("DYDX_V4_ALLOW_NON_MAINNET_INDEXER", default=False)
            indexer_url = normalize_and_validate_indexer_url(
                _get_str("DYDX_V4_INDEXER_URL", "https://indexer.dydx.trade"),
                allow_non_mainnet,
            )
        except ImportError:
            # exchange package not yet on the path (e.g. config.py imported
            # standalone before the project root is added to sys.path) —
            # fall back to the raw value; PaperExchange/DydxV4Adapter each
            # re-validate this themselves as a second line of defense.
            indexer_url = _get_str("DYDX_V4_INDEXER_URL", "https://indexer.dydx.trade")

        settings = Settings(
            dydx_v4_live_trading_enabled=live_trading_enabled,
            # PAPER_BALANCE: starting/simulated balance for PaperExchange.
            # Previously this was hardcoded to 15.0 in main.py's entrypoint
            # with no environment override at all — a real gap for a
            # deployment target (Render) that configures everything via
            # env vars. Also accepts the older PAPER_TRADING_BALANCE name
            # for backward compatibility with any existing deployment.
            paper_balance=_get_float("PAPER_BALANCE", _get_float("PAPER_TRADING_BALANCE", 15.0)),
            dydx_v4_mnemonic=mnemonic,
            dydx_v4_private_key=private_key,
            dydx_v4_node_url=_get_str("DYDX_V4_NODE_URL"),
            dydx_v4_indexer_url=indexer_url,
            dydx_v4_chain_id=_get_str("DYDX_V4_CHAIN_ID", "dydx-mainnet-1"),
            dydx_v4_usdc_denom=_get_str(
                "DYDX_V4_USDC_DENOM",
                "ibc/8E27BA2D5493AF5636760E354E46004562C46AB7EC0CC4C1CA14E9E20E2545B5",
            ),
            dydx_v4_subaccount_number=_get_int("DYDX_V4_SUBACCOUNT_NUMBER", 0),
            telegram_bot_token=_get_str("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_get_str("TELEGRAM_CHAT_ID"),
            # Which strategy TradingBot instantiates. "trend_pullback"
            # (macro EMA200 trend filter + EMA20/RSI pullback entry) is now
            # the default — 1m/5m raw EMA crossovers backtested with too
            # much noise/fee drag for positive expectancy. "trend_ema" is
            # kept available as the legacy option.
            strategy_type=_get_str("STRATEGY_TYPE", "regime_trend").strip().lower(),
            # Post-stop-out cooldown, in candles, before a same-direction
            # signal is allowed again for a symbol.
            strategy_cooldown_candles=_get_int("STRATEGY_COOLDOWN_CANDLES", 8),
            # Minimum ATR as a percentage of price required for a signal to
            # be considered actionable (filters dead/flat chop).
            strategy_min_atr_pct=_get_float("STRATEGY_MIN_ATR_PCT", 0.12),
            # trend_ema only: consecutive closed candles a crossover must
            # hold before it's treated as confirmed.
            strategy_confirmation_candles=_get_int("STRATEGY_CONFIRMATION_CANDLES", 2),
            strategy_n_donchian=_get_int("STRATEGY_N_DONCHIAN", 20),
            strategy_n_bb=_get_int("STRATEGY_N_BB", 20),
            strategy_bb_mult=_get_float("STRATEGY_BB_MULT", 2.0),
            strategy_n_percentile_lookback=_get_int("STRATEGY_N_PERCENTILE_LOOKBACK", 100),
            strategy_compression_percentile_threshold=_get_float("STRATEGY_COMPRESSION_PERCENTILE_THRESHOLD", 25.0),
            strategy_adx_min_for_entry=_get_float("STRATEGY_ADX_MIN_FOR_ENTRY", 20.0),
            strategy_n_vol_ma=_get_int("STRATEGY_N_VOL_MA", 20),
            strategy_volume_confirm_mult=_get_float("STRATEGY_VOLUME_CONFIRM_MULT", 1.2),
            strategy_atr_period=_get_int("STRATEGY_ATR_PERIOD", 14),
            strategy_atr_sl_mult=_get_float("STRATEGY_ATR_SL_MULT", 1.5),
            strategy_atr_tp_mult=_get_float("STRATEGY_ATR_TP_MULT", 3.0),
            strategy_max_hold_bars=_get_int("STRATEGY_MAX_HOLD_BARS", 48),
            # pullback EMA period, RSI settings, and whether RSI dipping
            # into oversold/overbought is required to confirm a pullback
            # (vs. relying on the EMA cross alone).
            strategy_ema_trend=_get_int("STRATEGY_EMA_TREND", 200),
            strategy_ema_pullback=_get_int("STRATEGY_EMA_PULLBACK", 20),
            strategy_rsi_period=_get_int("STRATEGY_RSI_PERIOD", 14),
            strategy_rsi_oversold=_get_float("STRATEGY_RSI_OVERSOLD", 40.0),
            strategy_rsi_overbought=_get_float("STRATEGY_RSI_OVERBOUGHT", 60.0),
            strategy_use_rsi_confirmation=parse_bool_env("STRATEGY_USE_RSI_CONFIRMATION", default=True),
            # ADX(period) trend-strength gate: only allow trend_pullback
            # entries when ADX exceeds this threshold — prevents entries
            # during low-volatility consolidation where pullback signals
            # tend to whipsaw.
            strategy_adx_period=_get_int("STRATEGY_ADX_PERIOD", 14),
            strategy_adx_threshold=_get_float("STRATEGY_ADX_THRESHOLD", 20.0),
            strategy_use_adx_filter=parse_bool_env("STRATEGY_USE_ADX_FILTER", default=True),
            # ATR-based take-profit: if set, TP = entry +/- N * ATR
            # directly, decoupled from risk_reward_ratio. Leave unset
            # (empty string) to keep the original SL-distance * RR
            # behavior.
            strategy_tp_atr_multiplier=(
                _get_float("STRATEGY_TP_ATR_MULTIPLIER", 0.0)
                if os.environ.get("STRATEGY_TP_ATR_MULTIPLIER", "").strip() else None
            ),
            # trend_pullback: dynamic (separate SL/TP) ATR multipliers and
            # volume-spike confirmation. Previously these were added directly
            # to strategies/trend_pullback.py's constructor defaults but
            # never wired through config.py/main.py — meaning they were NOT
            # actually configurable via environment variables at all, only
            # by editing the strategy file's hardcoded defaults directly.
            strategy_atr_multiplier_sl=_get_float("STRATEGY_ATR_MULTIPLIER_SL", 1.5),
            strategy_atr_multiplier_tp=_get_float("STRATEGY_ATR_MULTIPLIER_TP", 2.5),
            strategy_use_dynamic_atr_stops=parse_bool_env("STRATEGY_USE_DYNAMIC_ATR_STOPS", default=True),
            strategy_volume_ma_period=_get_int("STRATEGY_VOLUME_MA_PERIOD", 20),
            strategy_volume_spike_threshold=_get_float("STRATEGY_VOLUME_SPIKE_THRESHOLD", 1.1),
            strategy_use_volume_confirmation=parse_bool_env("STRATEGY_USE_VOLUME_CONFIRMATION", default=True),
            # Daily max drawdown (% of the day's starting equity) before
            # the RiskManager circuit breaker blocks all new orders until
            # the next UTC day.
            risk_max_daily_loss_pct=_get_float("RISK_MAX_DAILY_LOSS_PCT", 5.0),
            # Max leverage RiskManager will size a position at. NOTE: this
            # was previously hardcoded to 2.0 directly in main.py with no
            # config override at all — a real gap for anyone wanting a
            # different leverage (e.g. 3x) without editing source.
            risk_max_position_leverage=_get_float("RISK_MAX_POSITION_LEVERAGE", 2.0),
            # RISK_PER_TRADE (explicitly requested name) takes priority if
            # set; RISK_PER_TRADE_PCT is the original/legacy name and
            # remains supported for backward compatibility.
            risk_per_trade_pct=_get_float("RISK_PER_TRADE", _get_float("RISK_PER_TRADE_PCT", 0.25)),
            kill_max_daily_loss_pct=_get_float("KILL_MAX_DAILY_LOSS_PCT", 2.0),
            kill_max_consecutive_losses=_get_int("KILL_MAX_CONSECUTIVE_LOSSES", 3),
            kill_max_position_notional_pct=_get_float("KILL_MAX_POSITION_NOTIONAL_PCT", 5.0),
            kill_max_slippage_pct=_get_float("KILL_MAX_SLIPPAGE_PCT", 0.5),
            kill_max_orders_per_hour=_get_int("KILL_MAX_ORDERS_PER_HOUR", 10),
            kill_heartbeat_timeout_sec=_get_float("KILL_HEARTBEAT_TIMEOUT_SEC", 300.0),
            # Candle resolution for live trading + calibration. 1-minute
            # EMA crossovers proved too noisy/fee-heavy in backtesting;
            # default is now 5-minute candles.
            candle_resolution=normalize_candle_resolution(_get_str("CANDLE_RESOLUTION", "5MIN")),
            # Comma-separated ticker list, e.g. "BTC-USD,ETH-USD,SOL-USD".
            # Deduplicated, uppercased, order-preserving. Defaults to a
            # single symbol for backward compatibility with existing
            # single-asset deployments.
            tickers=_parse_tickers(_get_str("TICKERS", "BTC-USD,ETH-USD")),
            state_file_path=_get_str("STATE_FILE_PATH", "bot_state.json"),
            state_persistence_enabled=parse_bool_env("STATE_PERSISTENCE_ENABLED", default=True),
            log_level=_get_str("LOG_LEVEL", "INFO").upper(),
            dotenv_path=_DOTENV_PATH,
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        """
        Fail loudly and immediately on contradictory configuration instead
        of letting the bot silently fall back to paper mode. This is the
        direct fix for "I set DYDX_V4_LIVE_TRADING_ENABLED=true but it's
        still running PaperExchange" — that combination is now a hard
        startup error, not a silent downgrade.
        """
        if self.dydx_v4_live_trading_enabled:
            missing = []
            if not self.dydx_v4_mnemonic and not self.dydx_v4_private_key:
                missing.append("DYDX_V4_MNEMONIC or DYDX_V4_PRIVATE_KEY")
            if not self.dydx_v4_node_url:
                missing.append("DYDX_V4_NODE_URL")
            if not self.dydx_v4_indexer_url:
                missing.append("DYDX_V4_INDEXER_URL")
            if missing:
                raise ConfigError(
                    "DYDX_V4_LIVE_TRADING_ENABLED=true but the following "
                    f"required variable(s) are missing or empty: {missing}. "
                    "Refusing to start rather than silently falling back to "
                    "paper trading. Check your .env file is present at the "
                    f"project root (loaded from: {self.dotenv_path or 'nowhere — no .env found'}) "
                    "and that these variables are set."
                )
        else:
            if self.dydx_v4_mnemonic or self.dydx_v4_private_key:
                logger.warning(
                    "DYDX_V4_MNEMONIC or DYDX_V4_PRIVATE_KEY is set but DYDX_V4_LIVE_TRADING_ENABLED "
                    "is false/unset — running in PAPER mode. If you intended "
                    "to trade live, set DYDX_V4_LIVE_TRADING_ENABLED=true."
                )

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ConfigError(
                f"LOG_LEVEL={self.log_level!r} is invalid. Must be one of {sorted(valid_log_levels)}."
            )

        if self.strategy_cooldown_candles < 0:
            raise ConfigError("STRATEGY_COOLDOWN_CANDLES must be non-negative.")
        if self.strategy_min_atr_pct < 0:
            raise ConfigError("STRATEGY_MIN_ATR_PCT must be non-negative.")
        if self.strategy_confirmation_candles < 1:
            raise ConfigError("STRATEGY_CONFIRMATION_CANDLES must be at least 1.")
        if not (0 < self.risk_max_daily_loss_pct <= 100):
            raise ConfigError("RISK_MAX_DAILY_LOSS_PCT must be within (0, 100].")
        if self.risk_max_position_leverage <= 0:
            raise ConfigError("RISK_MAX_POSITION_LEVERAGE must be positive.")
        if self.risk_max_position_leverage > 5.0:
            logger.warning(
                "RISK_MAX_POSITION_LEVERAGE=%.1fx is unusually high for a small account — "
                "double-check this is intentional, not a typo.",
                self.risk_max_position_leverage,
            )
        if not (0 < self.risk_per_trade_pct <= 100):
            raise ConfigError("RISK_PER_TRADE_PCT must be within (0, 100].")

        if self.kill_max_daily_loss_pct >= self.risk_max_daily_loss_pct:
            raise ConfigError(
                f"KILL_MAX_DAILY_LOSS_PCT ({self.kill_max_daily_loss_pct}) must be strictly less than "
                f"RISK_MAX_DAILY_LOSS_PCT ({self.risk_max_daily_loss_pct}) to ensure the kill-switch triggers before or alongside daily risk limits."
            )

        if not self.tickers:
            raise ConfigError("TICKERS produced an empty list — at least one symbol is required.")
        for symbol in self.tickers:
            if "-" not in symbol:
                raise ConfigError(
                    f"TICKERS entry {symbol!r} doesn't look like a valid dYdX v4 market "
                    f"ticker (expected format like 'ETH-USD')."
                )

        if not self.state_file_path.strip():
            raise ConfigError("STATE_FILE_PATH must not be empty.")

        valid_strategy_types = {"volatility_expansion", "trend_pullback", "trend_ema", "regime_trend"}
        if self.strategy_type not in valid_strategy_types:
            raise ConfigError(
                f"STRATEGY_TYPE={self.strategy_type!r} is invalid. "
                f"Must be one of {sorted(valid_strategy_types)}."
            )
        if self.strategy_ema_trend <= 0 or self.strategy_ema_pullback <= 0 or self.strategy_rsi_period <= 0:
            raise ConfigError("STRATEGY_EMA_TREND, STRATEGY_EMA_PULLBACK, and STRATEGY_RSI_PERIOD must be positive.")
        if self.strategy_ema_pullback >= self.strategy_ema_trend:
            raise ConfigError("STRATEGY_EMA_PULLBACK must be strictly less than STRATEGY_EMA_TREND.")
        if not (0 < self.strategy_rsi_oversold < 50):
            raise ConfigError("STRATEGY_RSI_OVERSOLD must be within (0, 50).")
        if not (50 < self.strategy_rsi_overbought < 100):
            raise ConfigError("STRATEGY_RSI_OVERBOUGHT must be within (50, 100).")
        if self.strategy_adx_period <= 0:
            raise ConfigError("STRATEGY_ADX_PERIOD must be positive.")
        if self.strategy_adx_threshold < 0:
            raise ConfigError("STRATEGY_ADX_THRESHOLD must be non-negative.")
        if self.strategy_tp_atr_multiplier is not None and self.strategy_tp_atr_multiplier <= 0:
            raise ConfigError("STRATEGY_TP_ATR_MULTIPLIER must be positive when set.")
        if self.strategy_atr_multiplier_sl <= 0 or self.strategy_atr_multiplier_tp <= 0:
            raise ConfigError("STRATEGY_ATR_MULTIPLIER_SL and STRATEGY_ATR_MULTIPLIER_TP must be positive.")
        if self.strategy_volume_ma_period <= 0:
            raise ConfigError("STRATEGY_VOLUME_MA_PERIOD must be positive.")
        if self.strategy_volume_spike_threshold <= 0:
            raise ConfigError("STRATEGY_VOLUME_SPIKE_THRESHOLD must be positive.")
        if self.paper_balance <= 0:
            raise ConfigError("PAPER_BALANCE must be positive.")

    def summary(self) -> str:
        """Human-readable, secret-redacted summary for startup logs."""
        mnemonic_status = "SET" if self.dydx_v4_mnemonic else ("SET (private key)" if self.dydx_v4_private_key else "NOT SET")
        telegram_status = "SET" if (self.telegram_bot_token and self.telegram_chat_id) else "NOT SET"
        strategy_detail = (
            f"ema_trend={self.strategy_ema_trend}/ema_pullback={self.strategy_ema_pullback}/"
            f"rsi={self.strategy_rsi_period}"
            if self.strategy_type == "trend_pullback"
            else (
                f"n_donchian={self.strategy_n_donchian}/n_bb={self.strategy_n_bb}/"
                f"comp_thresh={self.strategy_compression_percentile_threshold}"
                if self.strategy_type == "volatility_expansion"
                else (
                    f"ema_fast=20/ema_slow=100/adx_min=22.0"
                    if self.strategy_type == "regime_trend"
                    else f"confirmation_candles={self.strategy_confirmation_candles}"
                )
            )
        )
        return (
            f"mode={'LIVE' if self.dydx_v4_live_trading_enabled else 'PAPER'} "
            f"| paper_balance=${self.paper_balance:.2f} "
            f"| tickers={','.join(self.tickers)} "
            f"| mnemonic={mnemonic_status} "
            f"| node_url={self.dydx_v4_node_url or '(unset)'} "
            f"| indexer_url={self.dydx_v4_indexer_url} "
            f"| subaccount={self.dydx_v4_subaccount_number} "
            f"| telegram={telegram_status} "
            f"| candle_resolution={self.candle_resolution} "
            f"| strategy={self.strategy_type} ({strategy_detail}) "
            f"| leverage={self.risk_max_position_leverage:.1f}x "
            f"| cooldown_candles={self.strategy_cooldown_candles} "
            f"| min_atr_pct={self.strategy_min_atr_pct:.3f}% "
            f"| max_daily_loss={self.risk_max_daily_loss_pct:.2f}% "
            f"| state_file={self.state_file_path} ({'enabled' if self.state_persistence_enabled else 'disabled'}) "
            f"| log_level={self.log_level}"
        )


# Single shared instance every other module should import.
settings = Settings.load()

logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
logger.info("Configuration loaded | %s", settings.summary())


if __name__ == "__main__":
    print(settings.summary())
    print(f"Loaded from: {settings.dotenv_path or 'no .env file found'}")