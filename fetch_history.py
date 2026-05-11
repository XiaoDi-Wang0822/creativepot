import ccxt
import pandas as pd
import time

def fetch_data(symbol='FET/USDT', timeframe='15m', limit=1000):
    # OKX is accessible and has FET-USDT, but we'll try to get as much as possible
    exchange = ccxt.okx()
    all_ohlcv = []

    # Let's try to find when it started on OKX
    # We go backwards from NOW (2026)
    end_time = exchange.milliseconds()

    while len(all_ohlcv) < 50000: # Increase limit to get more history
        try:
            # We want to go back far.
            # OKX fetch_ohlcv 'since' returns data AFTER since.
            # To get older data, we move since back.
            since = end_time - (limit * 15 * 60 * 1000)
            ohlcv = exchange.fetch_ohlcv('FET-USDT', timeframe, since=since, limit=limit)
            if not ohlcv or (all_ohlcv and ohlcv[0][0] == all_ohlcv[0][0]):
                break
            all_ohlcv = ohlcv + all_ohlcv
            end_time = ohlcv[0][0] - 1
            print(f"Candles fetched: {len(all_ohlcv)}")
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df.to_csv('fet_history.csv', index=False)
    print(f"Final history: {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

if __name__ == "__main__":
    fetch_data()
