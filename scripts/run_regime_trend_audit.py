"""
scripts/run_regime_trend_audit.py

Expanded IS/OOS (185 / 180 days) and full raw sensitivity matrix for BTC-USD and ETH-USD.
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

TAKER_FEE_PCT = 0.05
MAKER_FEE_PCT = 0.02
SLIPPAGE_PCT = 0.05


def run_regime_backtest(df: pd.DataFrame, symbol: str, strategy: RegimeTrendStrategy):
    fee_frac = TAKER_FEE_PCT / 100.0
    slippage_frac = SLIPPAGE_PCT / 100.0
    min_start = max(strategy.ema_slow, strategy.adx_period * 2, strategy.atr_period) + 10

    if len(df) <= min_start:
        return [], {}

    indicators = strategy.compute_indicators(df)
    trades = []
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
                applied_exit_fee = MAKER_FEE_PCT / 100.0 if exit_reason == "TAKE_PROFIT" else fee_frac
                entry_fee = fee_frac
                if side == "BUY":
                    raw_return = (exit_price - position["actual_entry"]) / position["actual_entry"]
                else:
                    raw_return = (position["actual_entry"] - exit_price) / position["actual_entry"]
                net_return = raw_return - entry_fee - applied_exit_fee

                trades.append(TradeResult(
                    side=side,
                    entry_time=position["entry_time"],
                    entry_price=position["actual_entry"],
                    exit_time=curr_ts,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    return_pct=net_return * 100.0,
                ))
                exit_breakdown[exit_reason] += 1
                if exit_reason in ("STOP_LOSS", "REGIME_INVALIDATION"):
                    cooldown_counter = strategy.cooldown_bars
                position = None

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
            atr = float(prev_closed["atr"])

            cross_up = (ema_fast_last <= ema_slow_last) and (ema_fast_prev > ema_slow_prev)
            cross_down = (ema_fast_last >= ema_slow_last) and (ema_fast_prev < ema_slow_prev)

            raw_side = None
            if cross_up and slope > 0:
                raw_side = "BUY"
            elif cross_down and slope < 0:
                raw_side = "SELL"
            elif ema_fast_prev > ema_slow_prev and slope > 0 and close_p <= ema_fast_prev * 1.002:
                raw_side = "BUY"
            elif ema_fast_prev < ema_slow_prev and slope < 0 and close_p >= ema_fast_prev * 0.998:
                raw_side = "SELL"

            if raw_side is not None:
                actual_entry = open_p * (1.0 + slippage_frac) if raw_side == "BUY" else open_p * (1.0 - slippage_frac)
                sl = actual_entry - strategy.atr_sl_mult * atr if raw_side == "BUY" else actual_entry + strategy.atr_sl_mult * atr
                tp = actual_entry + strategy.atr_tp_mult * atr if raw_side == "BUY" else actual_entry - strategy.atr_tp_mult * atr

                position = {
                    "symbol": symbol,
                    "side": raw_side,
                    "entry_time": curr_ts,
                    "actual_entry": actual_entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "entry_idx": i,
                }

    return trades, exit_breakdown


async def main():
    symbols = ["BTC-USD", "ETH-USD"]
    days = 365
    resolution = "1HOUR"
    indexer_url = settings.dydx_v4_indexer_url

    print("=" * 95)
    print("HYPOTHESIS #2: ADVANCED AUDIT (RAW SENSITIVITY MATRIX & 185D IS / 180D OOS)")
    print("=" * 95)

    for symbol in symbols:
        print(f"\nFetching {days} days of {resolution} candles for {symbol}...")
        df = await fetch_historical_candles(symbol, days, indexer_url, resolution=resolution)
        if len(df) < 200:
            continue

        # Expanded IS (185 days) / OOS (180 days) split
        # 185 days * 24 candles = 4440 candles
        is_candles = int(185 * 24)
        df_is = df.iloc[:is_candles].reset_index(drop=True)
        df_oos = df.iloc[is_candles:].reset_index(drop=True)

        strategy_base = RegimeTrendStrategy()
        trades_is, _ = run_regime_backtest(df_is, symbol, strategy_base)
        trades_oos, _ = run_regime_backtest(df_oos, symbol, strategy_base)

        perf_is = compute_performance(trades_is)
        perf_oos = compute_performance(trades_oos)

        print(f"\n[{symbol}] 185D IS / 180D OOS PERFORMANCE SPLIT:")
        print(f"  IS  (185 days): Trades={perf_is.total_trades} | WinRate={perf_is.win_rate_pct:.2f}% | PF={perf_is.profit_factor:.3f} | Return={perf_is.total_return_pct:+.2f}%")
        print(f"  OOS (180 days): Trades={perf_oos.total_trades} | WinRate={perf_oos.win_rate_pct:.2f}% | PF={perf_oos.profit_factor:.3f} | Return={perf_oos.total_return_pct:+.2f}% | MaxDD={perf_oos.max_drawdown_pct:.2f}%")

        print(f"\n[{symbol}] RAW SENSITIVITY MATRIX (ADX_MIN x SL_MULT):")
        print(f"{'ADX_MIN':<8} | {'SL_MULT':<8} | {'IS_PF':<8} | {'OOS_PF':<8} | {'OOS_Trd':<8} | {'OOS_Ret%':<10} | {'OOS_MaxDD%':<10}")
        print("-" * 75)

        for adx_m in [18, 22, 26]:
            for sl_m in [1.3, 1.5, 1.7]:
                strat_test = RegimeTrendStrategy(adx_min=adx_m, atr_sl_mult=sl_m)
                t_is, _ = run_regime_backtest(df_is, symbol, strat_test)
                t_oos, _ = run_regime_backtest(df_oos, symbol, strat_test)
                p_is = compute_performance(t_is)
                p_oos = compute_performance(t_oos)

                is_pf_s = f"{p_is.profit_factor:.3f}" if p_is.profit_factor is not None else "n/a"
                oos_pf_s = f"{p_oos.profit_factor:.3f}" if p_oos.profit_factor is not None else "n/a"
                print(f"{adx_m:<8} | {sl_m:<8} | {is_pf_s:<8} | {oos_pf_s:<8} | {p_oos.total_trades:<8} | {p_oos.total_return_pct:<+10.2f}% | {p_oos.max_drawdown_pct:<10.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
