import ccxt
import pandas as pd
import time

def fetch_data(symbol='FET-USDT', timeframe='15m', limit=1000):
    exchange = ccxt.okx()
    all_ohlcv = []
    end_time = exchange.milliseconds()

    while len(all_ohlcv) < 5000:
        try:
            since = end_time - (limit * 15 * 60 * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv = ohlcv + all_ohlcv
            end_time = ohlcv[0][0] - 1
            print(f"Total candles: {len(all_ohlcv)}")
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.to_csv('fet_15m.csv', index=False)

if __name__ == "__main__":
    fetch_data()
