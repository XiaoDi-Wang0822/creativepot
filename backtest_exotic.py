import pandas as pd
import numpy as np

def backtest(df, leverage=3.0):
    df = df.copy()

    # 1. Non-linear Smoothing: Jurik-like (EMA of EMA)
    df['ema1'] = df['close'].ewm(span=10).mean()
    df['jma'] = df['ema1'].ewm(span=10).mean()

    # 2. Kinetic Force (Volume * Price Acceleration)
    df['v'] = df['close'].diff(5)
    df['a'] = df['v'].diff(5)
    df['kinetic'] = df['volume'] * df['a']
    df['kinetic_z'] = (df['kinetic'] - df['kinetic'].rolling(100).mean()) / df['kinetic'].rolling(100).std()

    # 3. Efficiency Ratio
    df['er'] = df['close'].diff(14).abs() / df['close'].diff().abs().rolling(14).sum()

    equity = 100.0
    position = 0
    entry_price = 0
    trades = 0
    equity_history = []
    commission = 0.0006

    for i in range(100, len(df)):
        row = df.iloc[i]
        curr_close = row['close']

        if position != 0:
            pnl = (curr_close / entry_price - 1) * leverage if position == 1 else (entry_price / curr_close - 1) * leverage

            # Exit on Signal Flip or Hard Stop (10% equity loss = 3.3% price move)
            if (position == 1 and row['jma'] < df['jma'].iloc[i-1]) or (position == -1 and row['jma'] > df['jma'].iloc[i-1]) or pnl < -0.10:
                equity *= (1 + pnl - commission * leverage)
                position = 0
                trades += 1

        if position == 0:
            # Entry: High Efficiency + Kinetic Burst in trend direction
            if row['er'] > 0.6:
                if row['kinetic_z'] > 2.0 and row['jma'] > df['jma'].iloc[i-1]:
                    position = 1
                    entry_price = curr_close
                elif row['kinetic_z'] < -2.0 and row['jma'] < df['jma'].iloc[i-1]:
                    position = -1
                    entry_price = curr_close

        equity_history.append(equity)
        if equity <= 0: break

    roi = (equity - 100) / 100
    peak = pd.Series(equity_history).expanding().max()
    dd = (pd.Series(equity_history) - peak) / peak
    max_dd = dd.min()
    return roi, max_dd, trades

if __name__ == "__main__":
    df = pd.read_csv('fet_history.csv')
    roi, max_dd, trades = backtest(df)
    print(f"Exotic Logic Backtest: ROI={roi:.2%}, MaxDD={max_dd:.2%}, Trades={trades}")
