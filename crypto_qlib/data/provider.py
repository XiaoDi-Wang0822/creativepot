import os
import pandas as pd
import numpy as np
from .storage import HighPerformanceStorage

class DataProvider:
    def __init__(self, bin_dir='data/bin'):
        self.storage = HighPerformanceStorage(bin_dir)

    def get_instruments(self, market='top50'):
        if not os.path.exists(self.storage.bin_dir):
            return []
        return [d for d in os.listdir(self.storage.bin_dir) if os.path.isdir(os.path.join(self.storage.bin_dir, d))]

    def load_data(self, instruments, start_time, end_time, fields=['open', 'high', 'low', 'close', 'volume']):
        all_data = []
        for inst in instruments:
            df = self.storage.load_symbol(inst, start_time, end_time)
            if df is not None:
                df['instrument'] = inst
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        full_df = pd.concat(all_data)
        return full_df.set_index(['datetime', 'instrument']).sort_index()
