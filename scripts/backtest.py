"""
scripts/backtest.py

Autonomous backtest script for the Regime-Gated Trend Following strategy (`strategies/regime_trend.py`)
on BTC-USD and ETH-USD (1-hour resolution) over the last 6-12 months.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.regime_trend import RegimeTrendStrategy
from scripts.calibrate_strategy import fetch_historical_candles, compute_performance, TradeResult
from config import settings


def run_standalone_backtest(
    df: pd.DataFrame,
    symbol: str,
    strategy: RegimeTrendStrategy,
    initial_deposit: float = 1000.0,
    leverage: float = 3.0,
    risk_pct: float = 0.25,
    taker_fee_pct: float = 0.05,
    slippage_pct: float = 0.02,
):
    fee_frac = taker_fee_pct / 100.0
    slippage_frac = slippage_pct / 100.0
    min_start = max(strategy.ema_slow, strategy.adx_period * 2, strategy.atr_period) + 10

    if len(df) <= min_start:
        return [], initial_deposit, []

    indicators = strategy.compute_indicators(df)
    trades = []
    equity_curve = [initial_deposit]
    current_equity = initial_deposit
    position = None
    cooldown_counter = 0

    exit_breakdown = {
        "STOP_LOSS": 0,
        "TAKE_PROFIT": 0,
        "MAX_HOLD": 0,
        "REGIME_INVALIDATION": 0
    }

    for i in range(min_start, len(df)):
        candle = df.iloc[i]
        curr_ts = candle["timestamp"]
        open_p = float(candle["open"])
        high_p = float(candle["high"])
        low_p = float(candle["low"])
        close_p = float(candle["close"])

        prev_closed = indicators.iloc[i - 1]
        last_closed = indicators.iloc[i - 2]

        if cooldown_counter > 0:
            cooldown_counter -= 1

        # 1. Manage existing position
        if position is not None:
            side = position["side"]
            sl = position["stop_loss"]
            tp = position["take_profit"]
            entry_idx = position["entry_idx"]
            bars_held = i - entry_idx

            hit_sl = low_p <= sl if side == "BUY" else high_p >= sl
            hit_tp = high_p >= tp if side == "BUY" else low_p <= tp
            hit_max_hold = bars_held >= strategy.max_hold_bars

            slope_now = float(prev_closed["ema_slow_slope"])
            regime_invalidated = (side == "BUY" and slope_now < 0) or (side == "SELL" and slope_now > 0)

            exit_reason = None
            exit_price = 0.0

            if hit_sl:
                exit_reason, exit_price = "STOP_LOSS", sl
                exit_price = exit_price * (1.0 - slippage_frac) if side == "BUY" else exit_price * (1.0 + slippage_frac)
            elif hit_tp:
                exit_reason, exit_price = "TAKE_PROFIT", tp
            elif regime_invalidated:
                exit_reason, exit_price = "REGIME_INVALIDATION", open_p
                exit_price = exit_price * (1.0 - slippage_frac) if side == "BUY" else exit_price * (1.0 + slippage_frac)
            elif hit_max_hold:
                exit_reason, exit_price = "MAX_HOLD", open_p
                exit_price = exit_price * (1.0 - slippage_frac) if side == "BUY" else exit_price * (1.0 + slippage_frac)

            if exit_reason is not None:
                # Calculate return with fees & leverage
                entry_fee = fee_frac
                exit_fee = fee_frac
                
                if side == "BUY":
                    raw_return = (exit_price - position["actual_entry"]) / position["actual_entry"]
                else:
                    raw_return = (position["actual_entry"] - exit_price) / position["actual_entry"]

                net_return_pct = (raw_return * leverage - entry_fee - exit_fee) * 100.0
                dollar_pnl = position["allocated_margin"] * leverage * (net_return_pct / 100.0)
                
                current_equity += dollar_pnl
                equity_curve.append(current_equity)

                trades.append(TradeResult(
                    side=side,
                    entry_time=position["entry_time"],
                    entry_price=position["actual_entry"],
                    exit_time=curr_ts,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    return_pct=net_return_pct,
                ))
                exit_breakdown[exit_reason] += 1
                if exit_reason in ("STOP_LOSS", "REGIME_INVALIDATION"):
                    cooldown_counter = strategy.cooldown_bars
                position = None

        # 2. Check for new entry
        if position is None and cooldown_counter == 0:
            if any(pd.isna(prev_closed[col]) for col in ["ema_fast", "ema_slow", "atr", "adx", "ema_slow_slope"]):
                continue

            adx_window = indicators["adx"].iloc[i - strategy.adx_lookback_bars : i]
            if len(adx_window) < strategy.adx_lookback_bars or not (adx_window > strategy.adx_min).all():
                continue

            ema_fast_prev = float(prev_closed["ema_fast"])
            ema_slow_prev = float(prev_closed["ema_slow"])
            ema_fast_last = float(last_closed["ema_fast"])
            ema_slow_last = float(last_closed["ema_slow"])
            slope = float(prev_closed["ema_slow_slope"])
            close = float(prev_closed["close"])
            atr = float(prev_closed["atr"])

            cross_up = (ema_fast_last <= ema_slow_last) and (ema_fast_prev > ema_slow_prev)
            cross_down = (ema_fast_last >= ema_slow_last) and (ema_fast_prev < ema_slow_prev)

            side = None
            if cross_up and slope > 0:
                side = "BUY"
            elif cross_down and slope < 0:
                side = "SELL"
            elif ema_fast_prev > ema_slow_prev and slope > 0 and close <= ema_fast_prev * 1.002:
                side = "BUY"
            elif ema_fast_prev < ema_slow_prev and slope < 0 and close >= ema_fast_prev * 0.998:
                side = "SELL"

            if side is not None:
                actual_entry = open_p * (1.0 + slippage_frac) if side == "BUY" else open_p * (1.0 - slippage_frac)
                sl = actual_entry - strategy.atr_sl_mult * atr if side == "BUY" else actual_entry + strategy.atr_sl_mult * atr
                tp = actual_entry + strategy.atr_tp_mult * atr if side == "BUY" else actual_entry - strategy.atr_tp_mult * atr

                # Risk-based sizing
                risk_amount_usd = current_equity * (risk_pct / 100.0)
                sl_distance_pct = abs(actual_entry - sl) / actual_entry
                if sl_distance_pct > 0:
                    position_size_usd = risk_amount_usd / sl_distance_pct
                    allocated_margin = position_size_usd / leverage
                else:
                    allocated_margin = current_equity * 0.1

                position = {
                    "side": side,
                    "entry_time": curr_ts,
                    "actual_entry": actual_entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "entry_idx": i,
                    "allocated_margin": allocated_margin,
                }

    return trades, current_equity, equity_curve


async def main():
    print("=" * 80)
    print("AUTONOMOUS STRATEGY BACKTEST: scripts/backtest.py")
    print("Strategy: Regime-Gated Trend Following (strategies/regime_trend.py)")
    print("=" * 80)

    symbols = ["BTC-USD", "ETH-USD"]
    days = 180  # ~6 months of hourly candles
    resolution = "1HOUR"
    indexer_url = settings.dydx_v4_indexer_url

    initial_deposit = 1000.0
    leverage = 3.0
    risk_pct = 0.25
    taker_fee_pct = 0.05
    slippage_pct = 0.02

    strategy = RegimeTrendStrategy()

    all_combined_trades = []

    for symbol in symbols:
        print(f"\n[+] Fetching {days} days of {resolution} data for {symbol} from dYdX Indexer...")
        try:
            df = await fetch_historical_candles(symbol, days, indexer_url, resolution=resolution)
        except Exception as exc:
            print(f"[!] Failed to fetch data for {symbol}: {exc}")
            continue

        if len(df) < 100:
            print(f"[!] Insufficient data fetched for {symbol}: {len(df)} candles")
            continue

        print(f"    Fetched {len(df)} candles. Start: {df.iloc[0]['timestamp']}, End: {df.iloc[-1]['timestamp']}")

        trades, final_equity, equity_curve = run_standalone_backtest(
            df=df,
            symbol=symbol,
            strategy=strategy,
            initial_deposit=initial_deposit,
            leverage=leverage,
            risk_pct=risk_pct,
            taker_fee_pct=taker_fee_pct,
            slippage_pct=slippage_pct,
        )

        all_combined_trades.extend(trades)
        perf = compute_performance(trades)

        net_profit_usd = final_equity - initial_deposit
        net_profit_pct = (net_profit_usd / initial_deposit) * 100.0

        # Duration in hours
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600.0 for t in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        print(f"\n--------------------------------------------------------------------------------")
        print(f"RESULTS FOR {symbol}")
        print(f"--------------------------------------------------------------------------------")
        print(f"Total Trades             : {perf.total_trades}")
        print(f"Win Rate                 : {perf.win_rate_pct:.2f}% ({perf.wins} Wins / {perf.losses} Losses)")
        print(f"Profit Factor            : {perf.profit_factor:.3f}" if perf.profit_factor is not None else "Profit Factor: n/a")
        print(f"Net Profit ($)           : ${net_profit_usd:+,.2f} ({net_profit_pct:+.2f}%)")
        print(f"Max Drawdown (%)         : {perf.max_drawdown_pct:.2f}%")
        print(f"Average Trade Duration   : {avg_duration:.1f} hours")
        print(f"Expectancy ($ per trade) : ${perf.expectancy:,.2f}" if hasattr(perf, 'expectancy') else f"Expectancy: n/a")

    if all_combined_trades:
        combined_perf = compute_performance(all_combined_trades)
        total_durations = [(t.exit_time - t.entry_time).total_seconds() / 3600.0 for t in all_combined_trades]
        overall_avg_duration = sum(total_durations) / len(total_durations) if total_durations else 0.0

        print(f"\n" + "=" * 80)
        print("COMBINED PORTFOLIO PERFORMANCE (BTC-USD + ETH-USD)")
        print("=" * 80)
        print(f"Total Trades             : {combined_perf.total_trades}")
        print(f"Win Rate                 : {combined_perf.win_rate_pct:.2f}% ({combined_perf.wins} Wins / {combined_perf.losses} Losses)")
        print(f"Profit Factor            : {combined_perf.profit_factor:.3f}" if combined_perf.profit_factor is not None else "Profit Factor: n/a")
        print(f"Max Drawdown (%)         : {combined_perf.max_drawdown_pct:.2f}%")
        print(f"Average Trade Duration   : {overall_avg_duration:.1f} hours")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
