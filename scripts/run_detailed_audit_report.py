"""
scripts/run_detailed_audit_report.py

Extended diagnostic script to extract raw quarterly breakdowns, exit reason distributions
(TP / SL / Time-Stop), and filter suppression counts for BTC-USD and ETH-USD.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_strategy import fetch_historical_candles, compute_performance
from scripts.run_volatility_expansion_backtest import run_backtest_volatility_expansion
from config import settings


async def main():
    symbols = ["BTC-USD", "ETH-USD"]
    days = 365
    resolution = "1HOUR"
    indexer_url = settings.dydx_v4_indexer_url

    print("=" * 90)
    print("DETAILED CRO AUDIT REPORT: Volatility Expansion Breakout Strategy")
    print("Raw Quarterly Breakdown, Exit Reasons, and Filter Suppression Statistics")
    print("=" * 90)

    for symbol in symbols:
        print(f"\n[ASSET: {symbol}] Fetching {days} days of {resolution} candles...")
        df = await fetch_historical_candles(symbol, days, indexer_url, resolution=resolution)
        if len(df) < 200:
            print(f"Skipping {symbol}: insufficient candles ({len(df)})")
            continue

        # Split into 4 equal quarters
        min_time = df["timestamp"].min()
        max_time = df["timestamp"].max()
        total_duration = max_time - min_time
        q_duration = total_duration / 4

        print(f"\n--- {symbol}: Raw Quarterly Breakdown (365 Days) ---")
        print(f"{'Quarter':<32} | {'Trades':<6} | {'WinRate':<8} | {'ProfitFactor':<12} | {'MaxDD%':<8} | {'Return%':<8}")
        print("-" * 84)

        all_trades = []
        exit_reasons_count = {"TAKE_PROFIT": 0, "STOP_LOSS": 0, "TIME_STOP": 0}

        for q in range(4):
            q_start = min_time + q_duration * q
            q_end = min_time + q_duration * (q + 1)
            if q == 3:
                q_end = max_time + pd.Timedelta(seconds=1)
            q_df = df[(df["timestamp"] >= q_start) & (df["timestamp"] < q_end)].reset_index(drop=True)

            trades_q, funnel_q, _ = run_backtest_volatility_expansion(q_df, symbol)
            perf_q = compute_performance(trades_q)
            all_trades.extend(trades_q)

            for t in trades_q:
                exit_reasons_count[t.exit_reason] = exit_reasons_count.get(t.exit_reason, 0) + 1

            q_label = f"Q{q+1} ({q_start.strftime('%Y-%m-%d')} to {q_end.strftime('%Y-%m-%d')})"
            pf_str = f"{perf_q.profit_factor:.3f}" if perf_q.profit_factor is not None else "n/a"
            if pf_str == "inf":
                pf_str = "inf"

            print(f"{q_label:<32} | {perf_q.total_trades:<6} | {perf_q.win_rate_pct:>6.1f}% | {pf_str:>12} | {perf_q.max_drawdown_pct:>7.2f}% | {perf_q.total_return_pct:>7.2f}%")

        # Full year summary for symbol
        overall_perf = compute_performance(all_trades)
        overall_pf = f"{overall_perf.profit_factor:.3f}" if overall_perf.profit_factor else "n/a"
        print("-" * 84)
        print(f"{'FULL YEAR AGGREGATE':<32} | {overall_perf.total_trades:<6} | {overall_perf.win_rate_pct:>6.1f}% | {overall_pf:>12} | {overall_perf.max_drawdown_pct:>7.2f}% | {overall_perf.total_return_pct:>7.2f}%")

        # Exit reasons breakdown
        total_exits = sum(exit_reasons_count.values())
        print(f"\n--- {symbol}: Exit Reasons Breakdown ---")
        if total_exits > 0:
            for reason, count in exit_reasons_count.items():
                pct = (count / total_exits) * 100.0
                print(f"  - {reason:<12}: {count:3d} exits ({pct:5.1f}%)")
        else:
            print("  - No exits recorded (no trades executed)")

        # Full dataset funnel / filter breakdown
        _, full_funnel, _ = run_backtest_volatility_expansion(df, symbol)
        print(f"\n--- {symbol}: Filter Suppression Statistics (Full 365 Days) ---")
        print(f"  - Candles Evaluated           : {full_funnel.candles_evaluated}")
        print(f"  - Raw Breakout Setups         : {full_funnel.raw_crossovers}")
        print(f"  - Suppressed by Low Volatility: {full_funnel.suppressed_low_volatility}")
        print(f"  - Suppressed by Cooldown      : {full_funnel.suppressed_cooldown}")
        print(f"  - Actionable Trades Executed  : {full_funnel.actionable_signals}")
        print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
