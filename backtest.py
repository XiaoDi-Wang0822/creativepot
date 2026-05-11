import pandas as pd
import numpy as np

def backtest(df, leverage=3.0):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 1. Faster EMAs
    df['ma_fast'] = df['close'].rolling(50).mean()
    df['ma_slow'] = df['close'].rolling(200).mean()

    # 2. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    equity = 100.0
    position = 0
    entry_price = 0
    trades = 0
    equity_history = []
    commission = 0.0006

    # Date tracking
    start_date = df['timestamp'].iloc[200]
    end_date = df['timestamp'].iloc[-1]
    days = (end_date - start_date).total_seconds() / (24 * 3600)

    for i in range(200, len(df)):
        row = df.iloc[i]
        curr_close = row['close']

        if position == 1:
            if row['rsi'] > 60:
                pnl = (curr_close / entry_price - 1) * leverage
                equity *= (1 + pnl - commission * leverage)
                position = 0
                trades += 1
            elif (curr_close / entry_price - 1) * leverage < -0.045: # Adjusted stop
                equity *= (1 - 0.045 - commission * leverage)
                position = 0
                trades += 1
        elif position == -1:
            if row['rsi'] < 40:
                pnl = (entry_price / curr_close - 1) * leverage
                equity *= (1 + pnl - commission * leverage)
                position = 0
                trades += 1
            elif (entry_price / curr_close - 1) * leverage < -0.045:
                equity *= (1 - 0.045 - commission * leverage)
                position = 0
                trades += 1

        if position == 0:
            if row['ma_fast'] > row['ma_slow'] and row['rsi'] < 33: # Adjusted entry
                position = 1
                entry_price = curr_close
            elif row['ma_fast'] < row['ma_slow'] and row['rsi'] > 67:
                position = -1
                entry_price = curr_close

        equity_history.append(equity)
        if equity <= 0: break

    roi = (equity - 100) / 100
    peak = pd.Series(equity_history).expanding().max()
    dd = (pd.Series(equity_history) - peak) / peak
    max_dd = dd.min()
    daily_compounded_roi = (1 + roi)**(1/days) - 1 if days > 0 and (1+roi) > 0 else 0

    return roi, max_dd, trades, daily_compounded_roi, days

if __name__ == "__main__":
    df = pd.read_csv('fet_15m.csv')
    roi, max_dd, trades, d_roi, days = backtest(df)
    print(f"Dataset Days: {days:.2f}")
    print(f"Total ROI: {roi:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Total Trades: {trades}")
    print(f"Daily Compounded ROI: {d_roi:.2%}")
