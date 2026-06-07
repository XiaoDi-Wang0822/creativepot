import pandas as pd
import numpy as np

class FeatureExtractor:
    """
    Base class for feature extraction.
    """
    def __init__(self, data):
        self.data = data.copy()

    def add_alpha158(self):
        df = self.data
        for n in [5, 10, 20, 30, 60]:
            df[f'MA{n}'] = df.groupby('instrument', group_keys=False)['close'].transform(lambda x: x.rolling(n).mean())
            df[f'VMA{n}'] = df.groupby('instrument', group_keys=False)['volume'].transform(lambda x: x.rolling(n).mean())
        for n in [1, 5, 10, 20, 30, 60]:
            df[f'ROC{n}'] = df.groupby('instrument', group_keys=False)['close'].transform(lambda x: x.pct_change(n))
        for n in [5, 10, 20]:
            df[f'STD{n}'] = df.groupby('instrument', group_keys=False)['close'].transform(lambda x: x.rolling(n).std() / x.rolling(n).mean().replace(0, np.nan))

        def rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs.fillna(0)))
        df['RSI14'] = df.groupby('instrument', group_keys=False)['close'].transform(lambda x: rsi(x, 14))
        self.data = df
        return self

    def add_crypto_specific(self):
        df = self.data
        df['VolVelocity'] = df.groupby('instrument', group_keys=False)['volume'].transform(lambda x: x.pct_change(5))
        def get_corr_series(group):
            return group['close'].rolling(10).corr(group['volume'])
        corrs = []
        for inst, group in df.groupby('instrument'):
            c = get_corr_series(group)
            corrs.append(c)
        if corrs:
            df['PV_Corr'] = pd.concat(corrs)
        else:
            df['PV_Corr'] = np.nan
        self.data = df
        return self

    def add_labels(self, target_horizon=15):
        self.data['label'] = self.data.groupby('instrument', group_keys=False)['close'].transform(
            lambda x: x.shift(-target_horizon) / x.replace(0, np.nan) - 1
        )
        return self

    def get_features(self):
        # In dry run, we only want the last row which might have NaNs in some features if history is short,
        # but for training we dropna.
        # Let's keep it flexible.
        return self.data
