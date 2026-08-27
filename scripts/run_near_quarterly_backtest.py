"""
scripts/run_near_quarterly_backtest.py

Deep backtest of `trend_pullback` strategy on NEAR-USD for the last 365 days (1HOUR resolution),
split into 4 quarters (Walk-Forward), printing detailed performance metrics per quarter.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_strategy import (
    fetch_historical_candles,
    run_backtest_pullback,
    compute_performance,
)
from config import settings

async def main():
    symbol = "NEAR-USD"
    days = 365
    resolution = "1HOUR"
    indexer_url = settings.dydx_v4_indexer_url

    print(f"Fetching {days} days of {resolution} candles for {symbol} from dYdX Indexer...")
    df = await fetch_historical_candles(symbol, days, indexer_url, resolution=resolution)

    if len(df) < 100:
        print(f"Error: Not enough candles fetched ({len(df)}).")
        return

    print(f"Total candles fetched: {len(df)}")
    print(f"Start time: {df.iloc[0]['timestamp']}, End time: {df.iloc[-1]['timestamp']}")

    # Split df into 4 equal time quarters
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    total_duration = max_time - min_time
    quarter_duration = total_duration / 4

    quarters = []
    for i in range(4):
        q_start = min_time + quarter_duration * i
        q_end = min_time + quarter_duration * (i + 1)
        if i == 3:
            q_end = max_time + pd.Timedelta(seconds=1)
        q_df = df[(df["timestamp"] >= q_start) & (df["timestamp"] < q_end)].reset_index(drop=True)
        quarters.append((f"Q{i+1} ({q_start.strftime('%Y-%m-%d')} to {q_end.strftime('%Y-%m-%d')})", q_df))

    print("\n" + "=" * 80)
    print(f"QUARTERLY WALK-FORWARD BACKTEST: trend_ema on {symbol} (365 days, 1HOUR)")
    print("=" * 80)

    all_trades = []

    for q_name, q_df in quarters:
        if len(q_df) < 50:
            print(f"\n--- {q_name} ---")
            print(f"Insufficient candles ({len(q_df)}) for backtest.")
            continue

        trades_ema, _ = run_backtest_ema(
            q_df, symbol,
            fast_ema=settings.strategy_fast_ema,
            slow_ema=settings.strategy_slow_ema,
            atr_period=settings.strategy_atr_period,
            atr_multiplier=settings.strategy_atr_multiplier,
            risk_reward_ratio=settings.strategy_risk_reward_ratio,
            cooldown_candles=settings.strategy_cooldown_candles,
            min_atr_pct=settings.strategy_min_atr_pct,
            confirmation_candles=settings.strategy_confirmation_candles,
        )

        perf = compute_performance(trades_ema)
        all_trades.extend(trades_ema)

        pf_str = f"{perf.profit_factor:.3f}" if perf.profit_factor is not None else "n/a"
        if pf_str == "inf":
            pf_str = "inf"

        print(f"\n--- {q_name} ---")
        print(f"Candles in quarter: {len(q_df)}")
        print(f"Total Trades     : {perf.total_trades}")
        print(f"Win Rate         : {perf.win_rate_pct:.2f}% ({perf.wins}W / {perf.losses}L)")
        print(f"Profit Factor    : {pf_str}")
        print(f"Total Return     : {perf.total_return_pct:+.2f}%")
        print(f"Max Drawdown     : {perf.max_drawdown_pct:.2f}%")

    overall_perf = compute_performance(all_trades)
    print("\n" + "=" * 80)
    print(f"OVERALL 365-DAY PERFORMANCE (trend_ema on {symbol})")
    print("=" * 80)
    print(f"Total Trades     : {overall_perf.total_trades}")
    print(f"Win Rate         : {overall_perf.win_rate_pct:.2f}%")
    print(f"Profit Factor    : {overall_perf.profit_factor:.3f}")
    print(f"Total Return     : {overall_perf.total_return_pct:+.2f}%")
    print(f"Max Drawdown     : {overall_perf.max_drawdown_pct:.2f}%")

    # Overall full year summary
    overall_perf = compute_performance(all_trades)
    overall_pf_str = f"{overall_perf.profit_factor:.3f}" if overall_perf.profit_factor is not None else "n/a"

    print("\n" + "=" * 80)
    print("FULL YEAR 365-DAY AGGREGATE SUMMARY (NEAR-USD, 1HOUR)")
    print("=" * 80)
    print(f"Total Trades     : {overall_perf.total_trades}")
    print(f"Win Rate         : {overall_perf.win_rate_pct:.2f}% ({overall_perf.wins}W / {overall_perf.losses}L)")
    print(f"Profit Factor    : {overall_pf_str}")
    print(f"Max Drawdown     : {overall_perf.max_drawdown_pct:.3f}%")
    print(f"Total Return     : {overall_perf.total_return_pct:+.3f}%")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
