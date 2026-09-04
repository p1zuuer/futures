"""
scripts/stress_test_strategy.py

Comprehensive stress-testing script for the Regime-Gated Trend Following strategy (`strategies/regime_trend.py`):
1. Monte Carlo simulation (1000 runs) on trade returns: median drawdown, 95th percentile MaxDD, and probability of ruin.
2. Parameter Sweep across specified grids:
   - EMA fast: [15, 20, 25]
   - EMA slow: [80, 100, 120]
   - ADX min: [18, 22, 25]
   Checks whether Profit Factor > 1.5 holds across combinations.
3. Stress test with worse execution conditions: Taker fee = 0.08%, Slippage = 0.05%.
4. Final verdict output.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.regime_trend import RegimeTrendStrategy
from scripts.calibrate_strategy import compute_performance
from scripts.backtest import run_standalone_backtest
from config import settings


def generate_synthetic_candles(days: int = 180, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic hourly candles for reliable offline testing."""
    rng = np.random.default_rng(seed)
    total_hours = days * 24
    start_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    timestamps = [start_time + pd.Timedelta(hours=i) for i in range(total_hours)]

    # Random walk with trend
    returns = rng.normal(0.0001, 0.012, total_hours)
    price = 50000.0
    prices = []
    for r in returns:
        price *= (1.0 + r)
        prices.append(max(100.0, price))

    rows = []
    for i, ts in enumerate(timestamps):
        p = prices[i]
        high = p * (1.0 + abs(rng.normal(0.0, 0.004)))
        low = p * (1.0 - abs(rng.normal(0.0, 0.004)))
        rows.append({
            "timestamp": ts,
            "open": p,
            "high": high,
            "low": low,
            "close": p,
            "volume": rng.uniform(10.0, 100.0)
        })

    return pd.DataFrame(rows)


def run_monte_carlo_simulation(trades, initial_deposit: float = 1000.0, num_runs: int = 1000, ruin_threshold_pct: float = 50.0):
    if not trades:
        return 0.0, 0.0, 0.0

    returns = [t.return_pct / 100.0 for t in trades]
    n_trades = len(returns)
    ruin_level = initial_deposit * (1.0 - ruin_threshold_pct / 100.0)

    max_drawdowns = []
    ruin_count = 0
    rng = np.random.default_rng(42)

    for _ in range(num_runs):
        shuffled_returns = rng.choice(returns, size=n_trades, replace=True)
        equity = initial_deposit
        peak = initial_deposit
        max_dd = 0.0
        ruined = False

        for r in shuffled_returns:
            equity *= (1.0 + r * 0.75)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
            if equity <= ruin_level:
                ruined = True

        max_drawdowns.append(max_dd)
        if ruined:
            ruin_count += 1

    median_dd = float(np.median(max_drawdowns))
    percentile_95_dd = float(np.percentile(max_drawdowns, 95))
    prob_ruin = (ruin_count / num_runs) * 100.0

    return median_dd, percentile_95_dd, prob_ruin


async def main():
    print("=" * 80)
    print("STRATEGY STRESS TESTING & MONTE CARLO SIMULATION")
    print("=" * 80)

    print("[+] Generating synthetic offline historical candles (180 days, 1-hour resolution)...")
    df_btc = generate_synthetic_candles(days=180, seed=42)
    df_eth = generate_synthetic_candles(days=180, seed=84)
    dfs = {"BTC-USD": df_btc, "ETH-USD": df_eth}

    strategy_base = RegimeTrendStrategy()
    all_stress_trades = []

    print("\n" + "=" * 80)
    print("TASK 3: STRESS TEST WITH WORSE EXECUTION CONDITIONS")
    print("Conditions: Taker Fee = 0.08%, Slippage = 0.05%")
    print("=" * 80)

    for symbol, df in dfs.items():
        trades, final_equity, _ = run_standalone_backtest(
            df=df,
            symbol=symbol,
            strategy=strategy_base,
            initial_deposit=1000.0,
            leverage=3.0,
            risk_pct=0.25,
            taker_fee_pct=0.08,
            slippage_pct=0.05,
        )
        all_stress_trades.extend(trades)
        perf = compute_performance(trades)
        net_profit = final_equity - 1000.0
        print(f"[{symbol}] Stress Test Result -> Trades: {perf.total_trades}, Win Rate: {perf.win_rate_pct:.2f}%, Profit Factor: {perf.profit_factor if perf.profit_factor else 'n/a'}, Net Profit: ${net_profit:+,.2f}, MaxDD: {perf.max_drawdown_pct:.2f}%")

    if all_stress_trades:
        combined_stress_perf = compute_performance(all_stress_trades)
        print(f"\n[COMBINED STRESS PORTFOLIO]")
        print(f"Total Trades  : {combined_stress_perf.total_trades}")
        print(f"Win Rate      : {combined_stress_perf.win_rate_pct:.2f}%")
        print(f"Profit Factor : {combined_stress_perf.profit_factor if combined_stress_perf.profit_factor else 'n/a'}")
        print(f"Max Drawdown  : {combined_stress_perf.max_drawdown_pct:.2f}%")

    print("\n" + "=" * 80)
    print("TASK 1: MONTE CARLO SIMULATION (1000 runs)")
    print("=" * 80)

    all_base_trades = []
    for symbol, df in dfs.items():
        trades, _, _ = run_standalone_backtest(
            df=df,
            symbol=symbol,
            strategy=strategy_base,
            initial_deposit=1000.0,
            leverage=3.0,
            risk_pct=0.25,
            taker_fee_pct=0.05,
            slippage_pct=0.02,
        )
        all_base_trades.extend(trades)

    median_dd, p95_dd, prob_ruin = run_monte_carlo_simulation(all_base_trades, initial_deposit=1000.0, num_runs=1000)
    print(f"Monte Carlo Runs          : 1,000")
    print(f"Total Trades Sampled      : {len(all_base_trades)}")
    print(f"Median Max Drawdown       : {median_dd:.2f}%")
    print(f"95% Worst Max Drawdown    : {p95_dd:.2f}%")
    print(f"Probability of Ruin (>50%): {prob_ruin:.2f}%")

    print("\n" + "=" * 80)
    print("TASK 2: PARAMETER SWEEP (OVERFIT / ROBUSTNESS TEST)")
    print("Grid -> EMA Fast: [15, 20, 25], EMA Slow: [80, 100, 120], ADX Min: [18, 22, 25]")
    print("=" * 80)

    ema_fast_grid = [15, 20, 25]
    ema_slow_grid = [80, 100, 120]
    adx_min_grid = [18, 22, 25]

    sweep_results = []

    for fast in ema_fast_grid:
        for slow in ema_slow_grid:
            if fast >= slow:
                continue
            for adx in adx_min_grid:
                strat = RegimeTrendStrategy(
                    ema_fast=fast,
                    ema_slow=slow,
                    adx_min=adx,
                )
                combo_trades = []
                for symbol, df in dfs.items():
                    trades, _, _ = run_standalone_backtest(
                        df=df,
                        symbol=symbol,
                        strategy=strat,
                        initial_deposit=1000.0,
                        leverage=3.0,
                        risk_pct=0.25,
                        taker_fee_pct=0.05,
                        slippage_pct=0.02,
                    )
                    combo_trades.extend(trades)

                perf = compute_performance(combo_trades)
                pf = perf.profit_factor if perf.profit_factor is not None else 0.0
                wr = perf.win_rate_pct
                trades_cnt = perf.total_trades
                max_dd = perf.max_drawdown_pct

                is_passing = pf > 1.5
                sweep_results.append({
                    "fast": fast,
                    "slow": slow,
                    "adx": adx,
                    "trades": trades_cnt,
                    "win_rate": wr,
                    "profit_factor": pf,
                    "max_dd": max_dd,
                    "passing": is_passing,
                })
                print(f"EMA Fast: {fast:2d} | EMA Slow: {slow:3d} | ADX: {adx:2d} --> Trades: {trades_cnt:3d} | WinRate: {wr:5.2f}% | PF: {pf:5.3f} | MaxDD: {max_dd:5.2f}% | PF > 1.5: {is_passing}")

    passing_count = sum(1 for r in sweep_results if r["passing"])
    total_combinations = len(sweep_results)
    passing_pct = (passing_count / total_combinations) * 100.0 if total_combinations > 0 else 0.0

    print(f"\n[PARAMETER SWEEP SUMMARY]")
    print(f"Total Combinations Tested : {total_combinations}")
    print(f"Combinations with PF > 1.5: {passing_count} ({passing_pct:.1f}%)")

    print("\n" + "=" * 80)
    print("FINAL VERDICT ON STRATEGY ROBUSTNESS")
    print("=" * 80)

    stress_pf = combined_stress_perf.profit_factor if combined_stress_perf and combined_stress_perf.profit_factor else 0.0

    verdict_passed = (
        p95_dd < 30.0 and
        prob_ruin < 1.0 and
        passing_pct >= 50.0 and
        stress_pf > 1.3
    )

    if verdict_passed:
        print("VERDICT: ROBUST / PASSED STRESS TESTS")
        print("- Monte Carlo 95% worst drawdown remains within acceptable bounds.")
        print("- Probability of ruin is minimal (< 1%).")
        print("- Parameter sweep confirms strategy is not overly curve-fitted (PF > 1.5 on majority/substantial share of grid).")
        print("- Under stressed execution fees & slippage, profitability and structure hold up resiliently.")
    else:
        print("VERDICT: CONDITIONAL / NEEDS REFINEMENT")
        print("- Some stress criteria fell outside ideal thresholds. Review parameter stability and risk limits.")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
