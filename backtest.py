"""
backtest.py

Thin root-level entrypoint delegating to `scripts/calibrate_strategy.py`,
so `python backtest.py ...` works from the project root exactly like
`python scripts/calibrate_strategy.py ...` — same CLI, same behavior, no
duplicated logic.

Examples:
    python backtest.py --symbol ETH-USD --days 5 --resolution 5MIN
    python backtest.py --multi --resolution 1HOUR --days 60
    python backtest.py --multi --symbols BTC-USD,ETH-USD,SOL-USD --days 90

Run `python backtest.py --help` for the full flag list.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.calibrate_strategy import main

if __name__ == "__main__":
    asyncio.run(main())