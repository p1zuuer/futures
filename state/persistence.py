"""
state/persistence.py

Lightweight local JSON state store for restart safety (e.g. on Render,
where a container can restart at any time — deploys, crashes, host
maintenance). Persists exactly the state that otherwise lives only in
process memory and would be silently lost on restart:

    - Cooldown timestamps (TrendEmaStrategy/TrendPullbackStrategy)
    - Daily risk tracking + circuit-breaker status (RiskManager)
    - Open conditional order IDs pending stale-order cleanup (TradingBot,
      LIVE mode only — see main.py's `_cancel_stale_conditional_orders`)
    - Full simulated account state (PaperExchange only — LIVE mode's
      positions/balance live on dYdX itself, which is already the source
      of truth and needs no local mirror)

Design choices:
    - Single flat JSON file, atomic write (write to a temp file, then
      `os.replace()`) so a crash mid-write can never leave a corrupted,
      half-written state file behind.
    - Every load path is defensive: a missing, empty, or corrupted state
      file logs a warning and the bot starts fresh rather than crashing
      or refusing to boot. Restart safety should never mean "the bot
      can't start if its state file is damaged."
    - No external DB dependency — a single small JSON file is enough for
      a single-instance bot and keeps deployment trivial (no separate
      database service to provision on Render).

Author: Senior Python Async Developer
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("state_persistence")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

STATE_SCHEMA_VERSION = 1


class BotStateStore:
    """
    Reads and writes the bot's persisted state to a single local JSON
    file. Pure data in/out — this class knows nothing about
    TradingBot/RiskManager/strategy internals; `main.py` is responsible
    for building the dict to save and applying a loaded dict back onto
    live objects (keeps this module trivially testable in isolation).
    """

    def __init__(self, file_path: str = "bot_state.json") -> None:
        self.file_path = Path(file_path)

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load and return the persisted state dict, or None if the file
        doesn't exist, is empty, or fails to parse — in every one of
        those cases the caller should proceed with a fresh/default state
        rather than treating it as fatal.
        """
        if not self.file_path.exists():
            logger.info(
                "No state file found at %s — starting with fresh state "
                "(expected on first run).",
                self.file_path,
            )
            return None

        try:
            raw = self.file_path.read_text(encoding="utf-8")
            if not raw.strip():
                logger.warning("State file %s is empty — starting fresh.", self.file_path)
                return None
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to read/parse state file %s (%s) — starting fresh instead "
                "of refusing to boot. The corrupted file is left in place for "
                "inspection; it will be overwritten on the next successful save.",
                self.file_path, exc,
            )
            return None

        schema_version = data.get("schema_version")
        if schema_version != STATE_SCHEMA_VERSION:
            logger.warning(
                "State file %s has schema_version=%r (expected %d) — likely from "
                "an older bot version. Starting fresh rather than risking a "
                "mismatched/partial restore.",
                self.file_path, schema_version, STATE_SCHEMA_VERSION,
            )
            return None

        saved_at = data.get("saved_at", "unknown")
        logger.info("Loaded state from %s (saved_at=%s)", self.file_path, saved_at)
        return data

    def save(self, state: Dict[str, Any]) -> None:
        """
        Atomically write `state` to the JSON file. Adds `schema_version`
        and `saved_at` automatically. Writes to a temp file in the same
        directory and `os.replace()`s it into place, so a crash or power
        loss mid-write can never leave a half-written/corrupted file —
        readers always see either the old complete file or the new
        complete file, never a partial one.
        """
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **state,
        }

        directory = self.file_path.parent if str(self.file_path.parent) else Path(".")
        directory.mkdir(parents=True, exist_ok=True)

        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(directory), prefix=f".{self.file_path.name}.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, default=str)
                os.replace(tmp_path, self.file_path)
            except BaseException:
                # Clean up the temp file on any failure so we don't leak
                # stray .tmp files across repeated failed save attempts.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            # Persistence failures should never crash the trading loop —
            # log loudly and continue; the bot just loses restart-safety
            # for this tick, not correctness of the current session.
            logger.error("Failed to save state to %s: %s", self.file_path, exc)