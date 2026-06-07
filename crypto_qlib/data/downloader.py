import ccxt
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
from tqdm import tqdm

class DataDownloader:
    def __init__(self, exchange_id='binance', symbols=None, timeframe='1m'):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        self.symbols = symbols or []
        self.timeframe = timeframe

    def fetch_ohlcv(self, symbol, start_time, end_time):
        all_ohlcv = []
        since = self.exchange.parse8601(start_time)
        end_ts = self.exchange.parse8601(end_time)

        pbar = tqdm(total=(end_ts - since) // (60 * 1000 if self.timeframe == '1m' else 3600 * 1000))
        pbar.set_description(f"Downloading {symbol}")

        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                pbar.update(len(ohlcv))
                if len(ohlcv) < 1:
                    break
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                time.sleep(1)
        pbar.close()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[df['timestamp'] < end_ts]

    def download_all(self, start_time, end_time, save_dir='data/raw'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for symbol in self.symbols:
            safe_symbol = symbol.replace('/', '_')
            df = self.fetch_ohlcv(symbol, start_time, end_time)
            df.to_csv(os.path.join(save_dir, f"{safe_symbol}.csv"), index=False)
            print(f"Saved {symbol} to {save_dir}")

if __name__ == "__main__":
    # Example usage
    downloader = DataDownloader(symbols=['BTC/USDT', 'ETH/USDT'])
    # downloader.download_all('2023-01-01T00:00:00Z', '2023-01-02T00:00:00Z')
