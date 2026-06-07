import numpy as np
import pandas as pd
import os
import struct

class HighPerformanceStorage:
    """
    Qlib-like binary storage for OHLCV data.
    Each column is stored as a separate binary file for fast access.
    """
    COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'factor', 'tick_size']

    def __init__(self, bin_dir='data/bin'):
        self.bin_dir = bin_dir
        if not os.path.exists(self.bin_dir):
            os.makedirs(self.bin_dir)

    def _get_symbol_dir(self, symbol):
        safe_symbol = symbol.replace('/', '_')
        path = os.path.join(self.bin_dir, safe_symbol)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def convert_csv_to_bin(self, csv_path, symbol, tick_size=None):
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')

        symbol_dir = self._get_symbol_dir(symbol)

        # Save timestamps separately
        timestamps = df['timestamp'].values.astype(np.int64)
        timestamps.tofile(os.path.join(symbol_dir, 'timestamp.bin'))

        for col in self.COLUMNS:
            if col == 'tick_size':
                if tick_size is not None:
                    data = np.full(len(df), tick_size, dtype=np.float32)
                elif 'tick_size' in df.columns:
                    data = df['tick_size'].values.astype(np.float32)
                else:
                    data = np.full(len(df), 0.01, dtype=np.float32) # fallback
            elif col in df.columns:
                data = df[col].values.astype(np.float32)
            else:
                data = np.ones(len(df), dtype=np.float32)

            data.tofile(os.path.join(symbol_dir, f'{col}.bin'))

        print(f"Converted {symbol} to binary format in {symbol_dir}")

    def load_symbol(self, symbol, start_time=None, end_time=None):
        symbol_dir = self._get_symbol_dir(symbol)

        if not os.path.exists(os.path.join(symbol_dir, 'timestamp.bin')):
            return None

        timestamps = np.fromfile(os.path.join(symbol_dir, 'timestamp.bin'), dtype=np.int64)

        data = {}
        for col in self.COLUMNS:
            col_file = os.path.join(symbol_dir, f'{col}.bin')
            if os.path.exists(col_file):
                data[col] = np.fromfile(col_file, dtype=np.float32)

        df = pd.DataFrame(data)
        df['timestamp'] = timestamps
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        if start_time:
            df = df[df['datetime'] >= pd.to_datetime(start_time)]
        if end_time:
            df = df[df['datetime'] <= pd.to_datetime(end_time)]

        return df
