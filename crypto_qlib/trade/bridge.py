import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
from ..features.extractor import FeatureExtractor

class DryRunBridge:
    def __init__(self, exchange_id='mexc', symbols=['BTC/USDT'], model=None, ledger_path='ledger.json'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.symbols = symbols
        self.model = model
        self.ledger_path = ledger_path
        self._init_ledger()

    def _init_ledger(self):
        if not os.path.exists(self.ledger_path):
            ledger = {
                'balance': 10000.0,
                'positions': {s: 0.0 for s in self.symbols},
                'history': []
            }
            with open(self.ledger_path, 'w') as f:
                json.dump(ledger, f)

    def _load_ledger(self):
        with open(self.ledger_path, 'r') as f:
            return json.load(f)

    def _save_ledger(self, ledger):
        with open(self.ledger_path, 'w') as f:
            json.dump(ledger, f)

    def step(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dry run step...")
        ledger = self._load_ledger()

        # 1. Fetch current data for all symbols
        all_dfs = []
        current_prices = {}
        for symbol in self.symbols:
            # Fetch last 100 minutes to calculate features
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['instrument'] = symbol.replace('/', '_').replace(':', '_')
            all_dfs.append(df)
            current_prices[symbol] = ohlcv[-1][4]

        full_df = pd.concat(all_dfs).set_index(['datetime', 'instrument'])

        # 2. Extract features
        extractor = FeatureExtractor(full_df)
        # We don't add labels here as we are in dry run
        features = extractor.add_alpha158().add_crypto_specific().get_features()

        # 3. Get latest features per instrument
        latest_features = features.groupby(level='instrument').tail(1)

        # 4. Predict scores
        scores = self.model.predict(latest_features)
        latest_features = latest_features.copy()
        latest_features['score'] = scores

        # 5. Simple Strategy: Top-1 Long
        top_symbol_inst = latest_features['score'].idxmax()[1]
        top_symbol = [s for s in self.symbols if s.replace('/', '_').replace(':', '_') == top_symbol_inst][0]

        print(f"Top suggested symbol: {top_symbol} with score {latest_features.loc[(slice(None), top_symbol_inst), 'score'].iloc[0]}")

        # Simulate rebalancing
        portfolio_value = ledger['balance']
        for s, amt in ledger['positions'].items():
            portfolio_value += amt * current_prices[s]

        print(f"Current Portfolio Value: {portfolio_value:.2f}")

        # Rebalance: move everything to the top symbol
        # Sell others
        for s in self.symbols:
            if s != top_symbol and ledger['positions'][s] > 0:
                print(f"Dry Run: Selling {s}")
                ledger['balance'] += ledger['positions'][s] * current_prices[s] * (1 - 0.0006)
                ledger['positions'][s] = 0.0

        # Buy top
        if ledger['positions'][top_symbol] == 0:
            print(f"Dry Run: Buying {top_symbol}")
            buy_amt = ledger['balance'] / current_prices[top_symbol] * (1 - 0.0006)
            ledger['positions'][top_symbol] = buy_amt
            ledger['balance'] = 0.0

        ledger['history'].append({
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio_value': portfolio_value
        })

        self._save_ledger(ledger)
        print("Dry run step completed.")
