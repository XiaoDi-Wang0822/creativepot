import os
import time
from .downloader import DataDownloader
from .storage import HighPerformanceStorage
from ..utils.cmc_helper import get_top_symbols

class DataPipeline:
    def __init__(self, bin_dir='data/bin', raw_dir='data/raw', exchange_id='okx'):
        self.bin_dir = bin_dir
        self.raw_dir = raw_dir
        self.exchange_id = exchange_id
        self.storage = HighPerformanceStorage(bin_dir=bin_dir)

    def run(self, limit=50, timeframe='1m', start_time='2024-01-01T00:00:00Z', end_time='2024-01-02T00:00:00Z'):
        """
        Full pipeline: Fetch eligible symbols -> Download -> Convert to binary
        """
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)

        print(f"Step 1: Fetching top {limit} symbols from {self.exchange_id} with >6 months history...")
        symbols = get_top_symbols(exchange_id=self.exchange_id, limit=limit)
        print(f"Found {len(symbols)} eligible symbols: {symbols}")

        downloader = DataDownloader(exchange_id=self.exchange_id, symbols=symbols, timeframe=timeframe)

        for symbol in symbols:
            print(f"Step 2: Downloading {symbol}...")
            safe_symbol = symbol.replace('/', '_').replace(':', '_')
            df = downloader.fetch_ohlcv(symbol, start_time, end_time)

            if df.empty:
                print(f"Warning: No data for {symbol}")
                continue

            csv_path = os.path.join(self.raw_dir, f"{safe_symbol}.csv")
            df.to_csv(csv_path, index=False)

            print(f"Step 3: Converting {symbol} to high-performance binary...")
            try:
                market_info = downloader.exchange.market(symbol)
                tick_size = market_info['precision']['price']
                if isinstance(tick_size, int):
                    tick_size = 10 ** (-tick_size)
                else:
                    tick_size = float(tick_size)
            except:
                tick_size = 0.01

            self.storage.convert_csv_to_bin(csv_path, symbol, tick_size=tick_size)

        print("Data Pipeline Completed.")
