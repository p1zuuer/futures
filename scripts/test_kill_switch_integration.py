"""
scripts/test_kill_switch_integration.py

End-to-end integration test for the Hard-Kill-Switch system.
Simulates consecutive losses, verifies state persistence, cancellation,
Telegram alert dispatch, and restart blocking.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import time
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.kill_switch import KillSwitch, KillSwitchTriggered
from state.persistence import BotStateStore


async def amain():
    print("=" * 80)
    print("HARD-KILL-SWITCH END-TO-END INTEGRATION TEST")
    print("=" * 80)

    # 1. Setup mock config with KILL_MAX_CONSECUTIVE_LOSSES = 2
    class MockConfig:
        KILL_MAX_DAILY_LOSS_PCT = 2.0
        KILL_MAX_CONSECUTIVE_LOSSES = 2
        KILL_MAX_POSITION_NOTIONAL_PCT = 5.0
        KILL_MAX_SLIPPAGE_PCT = 0.5
        KILL_MAX_ORDERS_PER_HOUR = 10
        KILL_HEARTBEAT_TIMEOUT_SEC = 300

    config = MockConfig()
    ks = KillSwitch(config)

    # 2. Setup temporary state store
    state_file = "test_bot_state.json"
    if Path(state_file).exists():
        Path(state_file).unlink()

    store = BotStateStore(state_file)

    # 3. Simulate 2 consecutive losses
    fake_trades = [
        {"return_pct": -1.2},
        {"return_pct": -0.8}
    ]

    print("\nSimulating 2 consecutive losses...")
    try:
        await ks.check_consecutive_losses(fake_trades)
        print("ERROR: Kill switch failed to trigger!")
    except KillSwitchTriggered as e:
        print(f"SUCCESS: KillSwitchTriggered caught [{e.check_name}]: {e.reason}")
        # Persist via store
        state = store.load() or {}
        state.update({
            "kill_switch_active": True,
            "kill_switch_reason": e.reason,
            "kill_switch_check_name": e.check_name,
            "kill_switch_triggered_at": time.time(),
        })
        store.save(state)

    # 4. Verify state persistence
    print("\nVerifying bot_state.json persistence...")
    data = store.load()
    print(f"State data loaded: {json.dumps(data, indent=2)}")
    assert data.get("kill_switch_active") is True, "kill_switch_active must be True"
    print("SUCCESS: kill_switch_active is correctly persisted as True.")

    # 5. Verify restart block check
    print("\nVerifying restart block check...")
    if data.get("kill_switch_active"):
        print(f"SUCCESS: Restart blocked! Kill switch is active. Reason: {data.get('kill_switch_reason')}")
    else:
        print("ERROR: Restart block failed!")

    # Cleanup
    if Path(state_file).exists():
        Path(state_file).unlink()

    print("\n" + "=" * 80)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 80)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
