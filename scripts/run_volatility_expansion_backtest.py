"""
scripts/run_volatility_expansion_backtest.py

Walk-Forward backtest and quarterly breakdown script for Volatility Expansion Breakout strategy
on BTC-USD and ETH-USD (1H candles, last 365 days).
Split:
  - IS (In-Sample): First 273 days (75%) for parameter calibration / inspection.
  - OOS (Out-of-Sample): Last 92 days (25%) frozen test.
  - Quarterly breakdown: 4 quarters of ~90 days each.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_strategy import (
    fetch_historical_candles,
    compute_performance,
    TradeResult,
    SignalFunnel,
    PerformanceStats,
    TAKER_FEE_PCT,
)
from strategies.volatility_expansion import VolatilityExpansionStrategy
from config import settings


def run_backtest_volatility_expansion(
    df: pd.DataFrame,
    symbol: str,
    n_donchian: int = 20,
    n_bb: int = 20,
    bb_mult: float = 2.0,
    n_percentile_lookback: int = 100,
    compression_percentile_threshold: float = 25.0,
    adx_period: int = 14,
    adx_min_for_entry: float = 20.0,
    n_vol_ma: int = 20,
    volume_confirm_mult: float = 1.2,
    atr_period: int = 14,
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 3.0,
    max_hold_bars: int = 48,
    cooldown_candles: int = 5,
    min_atr_pct: float = 0.05,
) -> tuple[list[TradeResult], SignalFunnel, VolatilityExpansionStrategy]:
    """
    Walk-forward, single-position backtest engine for VolatilityExpansionStrategy.
    Applies O(1) vectorized indicator lookups and respects TAKER fees + SLIPPAGE (0.05% on entry & stop-market exit).
    """
    strategy_logger = logging_level = None  # suppress strategy verbose logs if needed
    strategy = VolatilityExpansionStrategy(
        n_donchian=n_donchian,
        n_bb=n_bb,
        bb_mult=bb_mult,
        n_percentile_lookback=n_percentile_lookback,
        compression_percentile_threshold=compression_percentile_threshold,
        adx_period=adx_period,
        adx_min_for_entry=adx_min_for_entry,
        n_vol_ma=n_vol_ma,
        volume_confirm_mult=volume_confirm_mult,
        atr_period=atr_period,
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
        max_hold_bars=max_hold_bars,
        cooldown_candles=cooldown_candles,
        min_atr_pct=min_atr_pct,
    )

    indicators = strategy.calculate_indicators(df)
    funnel = SignalFunnel()
    trades: list[TradeResult] = []
    position: Optional[dict] = None
    fee_frac = TAKER_FEE_PCT / 100.0
    slippage_frac = 0.0005  # 0.05% slippage on entry and stop-market exit
    min_start = max(n_donchian, n_bb, n_percentile_lookback, adx_period * 2, n_vol_ma, atr_period) + 10

    for i in range(min_start, len(df)):
        current_candle = df.iloc[i]
        bars_held = 0 if position is None else (i - position["entry_idx"])

        if position is not None:
            side = position["side"]
            sl, tp = position["sl"], position["tp"]
            
            hit_sl = (current_candle["low"] <= sl) if side == "BUY" else (current_candle["high"] >= sl)
            hit_tp = (current_candle["high"] >= tp) if side == "BUY" else (current_candle["low"] <= tp)
            time_stop = bars_held >= max_hold_bars

            exit_reason = None
            exit_price = None

            if hit_sl:
                exit_reason, exit_price = "STOP_LOSS", sl
                # apply slippage on stop exit (taker)
                exit_price = exit_price * (1.0 - slippage_frac) if side == "BUY" else exit_price * (1.0 + slippage_frac)
            elif hit_tp:
                exit_reason, exit_price = "TAKE_PROFIT", tp
                # TP is maker limit order -> no slippage, and maker fee (0.02%)
                pass
            elif time_stop:
                exit_reason, exit_price = "TIME_STOP", current_candle["close"]
                # time stop executed as market (taker)
                exit_price = exit_price * (1.0 - slippage_frac) if side == "BUY" else exit_price * (1.0 + slippage_frac)

            if exit_reason is not None:
                # determine fee rate: maker (0.02%) for TP, taker (0.05%) for SL / Time Stop
                applied_exit_fee = 0.0002 if exit_reason == "TAKE_PROFIT" else fee_frac
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
                    exit_time=current_candle["timestamp"],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    return_pct=net_return * 100.0,
                ))

                if exit_reason == "STOP_LOSS":
                    strategy.record_stop_out(symbol, side, current_candle["timestamp"])

                position = None
            else:
                continue

        funnel.candles_evaluated += 1

        # Check signal at i using indicators computed up to i-1
        prev_closed = indicators.iloc[i - 1]
        last_closed = indicators.iloc[i - 2] if i >= 2 else prev_closed

        # Verify required columns are present and not NaN
        required_fields = [
            "donchian_high", "donchian_low", "bb_width_percentile",
            "adx", "volume_ma", "atr"
        ]
        if indicators.iloc[i - 1][required_fields].isna().any():
            funnel.insufficient_data_skips += 1
            continue

        close_now = float(df.iloc[i - 1]["close"])
        volume_now = float(df.iloc[i - 1]["volume"])
        atr_value = float(indicators.iloc[i - 1]["atr"])

        compression_flag = float(indicators.iloc[i - 2]["bb_width_percentile"]) <= strategy.compression_percentile_threshold
        breakout_high = float(indicators.iloc[i - 2]["donchian_high"])
        breakout_low = float(indicators.iloc[i - 2]["donchian_low"])
        breakout_long = close_now > breakout_high
        breakout_short = close_now < breakout_low

        volume_ma_prev = float(indicators.iloc[i - 2]["volume_ma"])
        volume_flag = volume_ma_prev > 0 and volume_now > (strategy.volume_confirm_mult * volume_ma_prev)

        adx_now = float(indicators.iloc[i - 1]["adx"])
        adx_prev = float(indicators.iloc[i - 2]["adx"])
        adx_flag = adx_now > strategy.adx_min_for_entry and adx_now > adx_prev

        long_setup = compression_flag and breakout_long and volume_flag and adx_flag
        short_setup = compression_flag and breakout_short and volume_flag and adx_flag

        funnel.raw_crossovers += 1
        funnel.confirmed_crossovers += 1

        raw_side = "BUY" if long_setup else ("SELL" if short_setup else "HOLD")
        if raw_side == "HOLD":
            continue

        entry_price = close_now
        atr_pct = (atr_value / entry_price * 100.0) if entry_price > 0 else 0.0
        candle_ts = pd.Timestamp(indicators.iloc[i - 1]["timestamp"])

        if atr_pct < strategy.min_atr_pct:
            funnel.suppressed_low_volatility += 1
            continue

        if strategy._cooldown_active(symbol, raw_side, candle_ts, pd.Timedelta(hours=1)):
            funnel.suppressed_cooldown += 1
            continue

        funnel.actionable_signals += 1

        # Apply slippage on entry (taker market order)
        actual_entry = entry_price * (1.0 + slippage_frac) if raw_side == "BUY" else entry_price * (1.0 - slippage_frac)

        sl_distance = atr_value * strategy.atr_sl_mult
        tp_distance = atr_value * strategy.atr_tp_mult

        if raw_side == "BUY":
            stop_loss = actual_entry - sl_distance
            take_profit = actual_entry + tp_distance
        else:
            stop_loss = actual_entry + sl_distance
            take_profit = actual_entry - tp_distance

        position = {
            "side": raw_side,
            "entry_idx": i - 1,
            "entry_time": candle_ts,
            "actual_entry": actual_entry,
            "sl": stop_loss,
            "tp": take_profit,
        }

    return trades, funnel, strategy


async def main():
    symbols = ["BTC-USD", "ETH-USD"]
    days = 365
    resolution = "1HOUR"
    indexer_url = settings.dydx_v4_indexer_url

    print("=" * 84)
    print("WALK-FORWARD BACKTEST & QUARTERLY BREAKDOWN: Volatility Expansion Breakout Strategy")
    print("Assets: BTC-USD, ETH-USD | Timeframe: 1H | Period: Last 365 Days")
    print("=" * 84)

    for symbol in symbols:
        print(f"\nFetching {days} days of {resolution} candles for {symbol}...")
        df = await fetch_historical_candles(symbol, days, indexer_url, resolution=resolution)
        if len(df) < 200:
            print(f"Error: Insufficient candles fetched for {symbol} ({len(df)}).")
            continue

        print(f"Fetched {len(df)} candles for {symbol}. Range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")

        # Split: 75% IS (first 273 days), 25% OOS (last 92 days)
        split_idx = int(len(df) * 0.75)
        df_is = df.iloc[:split_idx].reset_index(drop=True)
        df_oos = df.iloc[split_idx:].reset_index(drop=True)

        print(f"\n--- IN-SAMPLE (IS) PERFORMANCE: {symbol} ({len(df_is)} candles, ~75%) ---")
        trades_is, funnel_is, _ = run_backtest_volatility_expansion(df_is, symbol)
        perf_is = compute_performance(trades_is)
        print(f"Total Trades : {perf_is.total_trades}")
        print(f"Win Rate     : {perf_is.win_rate_pct:.2f}% ({perf_is.wins}W / {perf_is.losses}L)")
        print(f"Profit Factor: {perf_is.profit_factor:.3f}" if perf_is.profit_factor else "Profit Factor: n/a")
        print(f"Max Drawdown : {perf_is.max_drawdown_pct:.3f}%")
        print(f"Total Return : {perf_is.total_return_pct:+.3f}%")

        print(f"\n--- OUT-OF-SAMPLE (OOS) PERFORMANCE (FROZEN): {symbol} ({len(df_oos)} candles, ~25%) ---")
        trades_oos, funnel_oos, _ = run_backtest_volatility_expansion(df_oos, symbol)
        perf_oos = compute_performance(trades_oos)
        print(f"Total Trades : {perf_oos.total_trades}")
        print(f"Win Rate     : {perf_oos.win_rate_pct:.2f}% ({perf_oos.wins}W / {perf_oos.losses}L)")
        print(f"Profit Factor: {perf_oos.profit_factor:.3f}" if perf_oos.profit_factor else "Profit Factor: n/a")
        print(f"Max Drawdown : {perf_oos.max_drawdown_pct:.3f}%")
        print(f"Total Return : {perf_oos.total_return_pct:+.3f}%")

        # Quarterly breakdown (4 quarters of ~90 days)
        min_time = df["timestamp"].min()
        max_time = df["timestamp"].max()
        total_duration = max_time - min_time
        q_duration = total_duration / 4

        print(f"\n--- QUARTERLY BREAKDOWN: {symbol} ---")
        for q in range(4):
            q_start = min_time + q_duration * q
            q_end = min_time + q_duration * (q + 1)
            if q == 3:
                q_end = max_time + pd.Timedelta(seconds=1)
            q_df = df[(df["timestamp"] >= q_start) & (df["timestamp"] < q_end)].reset_index(drop=True)

            trades_q, _, _ = run_backtest_volatility_expansion(q_df, symbol)
            perf_q = compute_performance(trades_q)
            pf_str = f"{perf_q.profit_factor:.3f}" if perf_q.profit_factor else "n/a"
            print(f"  Q{q+1} ({q_start.strftime('%Y-%m-%d')} to {q_end.strftime('%Y-%m-%d')}): "
                  f"Trades={perf_q.total_trades}, WR={perf_q.win_rate_pct:.1f}%, "
                  f"PF={pf_str}, DD={perf_q.max_drawdown_pct:.2f}%, Ret={perf_q.total_return_pct:+.2f}%")

        print("-" * 84)


if __name__ == "__main__":
    asyncio.run(main())
