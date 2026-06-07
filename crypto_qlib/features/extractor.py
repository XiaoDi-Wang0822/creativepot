import pandas as pd
import numpy as np

class FeatureExtractor:
    """
    Base class for feature extraction.
    Porting Qlib's Alpha158 and Alpha360 style factors.
    """
    def __init__(self, data):
        self.data = data.copy()

    def add_alpha158(self):
        """
        Implementation of subset of Alpha158 factors suitable for crypto.
        Focusing on price and volume patterns.
        """
        df = self.data
        # Simple moving averages
        for n in [5, 10, 20, 30, 60]:
            df[f'MA{n}'] = df.groupby('instrument', group_keys=False)['close'].transform(lambda x: x.rolling(n).mean())
            df[f'VMA{n}'] = df.groupby('instrument', group_keys=False)['volume'].transform(lambda x: x.rolling(n).mean())

        # Price change (Return)
        for n in [1, 5, 10, 20, 30, 60]:
            df[f'ROC{n}'] = df.groupby('instrument', group_keys=False)['close'].transform(lambda x: x.pct_change(n))

        # Volatility
        for n in [5, 10, 20]:
            df[f'STD{n}'] = df.groupby('instrument', group_keys=False)['close'].transform(lambda x: x.rolling(n).std() / x.rolling(n).mean().replace(0, np.nan))

        # RSI (Simplified)
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
        """
        Adding crypto-specific features.
        """
        df = self.data
        # Volume Velocity
        df['VolVelocity'] = df.groupby('instrument', group_keys=False)['volume'].transform(lambda x: x.pct_change(5))

        # Price-Volume Correlation - Using transform-like logic for better compatibility
        def get_corr_series(group):
            return group['close'].rolling(10).corr(group['volume'])

        # We'll calculate it per instrument and then map back or use a loop for safety in pandas 3.0
        corrs = []
        for inst, group in df.groupby('instrument'):
            c = get_corr_series(group)
            corrs.append(c)

        df['PV_Corr'] = pd.concat(corrs)

        self.data = df
        return self

    def add_labels(self, target_horizon=15):
        """
        Label: Future return over target_horizon minutes.
        """
        self.data['label'] = self.data.groupby('instrument', group_keys=False)['close'].transform(
            lambda x: x.shift(-target_horizon) / x.replace(0, np.nan) - 1
        )
        return self

    def get_features(self):
        # Drop rows with NaN caused by rolling windows
        return self.data.dropna()
