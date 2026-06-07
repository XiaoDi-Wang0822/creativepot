import pandas as pd
import numpy as np
import os
from crypto_qlib.data.storage import HighPerformanceStorage
from crypto_qlib.data.provider import DataProvider
from crypto_qlib.features.extractor import FeatureExtractor
from crypto_qlib.models.wrapper import LGBModel, GRUModel
from crypto_qlib.backtest.engine import SimpleBacktester
from crypto_qlib.analysis.analyzer import Analyzer

def test_full_pipeline():
    # Setup dummy data
    bin_dir = 'data/test_bin'
    os.makedirs('data/raw', exist_ok=True)
    dates = pd.date_range('2023-01-01', periods=200, freq='1min')
    for symbol in ['BTC_USDT', 'ETH_USDT']:
        df = pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],
            'datetime': dates,
            'open': np.random.randn(200) + 20000,
            'high': np.random.randn(200) + 20100,
            'low': np.random.randn(200) + 19900,
            'close': np.random.randn(200) + 20000,
            'volume': np.random.rand(200) * 100,
            'tick_size': np.full(200, 0.1)
        })
        csv_path = f'data/raw/{symbol}.csv'
        df.to_csv(csv_path, index=False)
        storage = HighPerformanceStorage(bin_dir=bin_dir)
        storage.convert_csv_to_bin(csv_path, symbol.replace('_', '/'))

    # Pipeline
    provider = DataProvider(bin_dir=bin_dir)
    data = provider.load_data(['BTC_USDT', 'ETH_USDT'], '2023-01-01 00:00:00', '2023-01-01 03:00:00')
    assert not data.empty

    extractor = FeatureExtractor(data)
    features = extractor.add_alpha158().add_crypto_specific().add_labels().get_features()
    assert not features.empty

    train_df = features.iloc[:100]
    test_df = features.iloc[100:]

    # Test LGB
    model = LGBModel()
    model.fit(train_df)
    preds = model.predict(test_df)
    test_df_lgb = test_df.copy()
    test_df_lgb['score'] = preds

    backtester = SimpleBacktester()
    report_df = backtester.run(test_df_lgb[['score']], data, topk=1)
    assert not report_df.empty

    # Test GRU
    gru_model = GRUModel(input_dim=train_df.shape[1]-1, epochs=2)
    gru_model.fit(train_df)
    preds_gru = gru_model.predict(test_df)
    test_df_gru = test_df.copy()
    test_df_gru['score'] = preds_gru

    report_df_gru = backtester.run(test_df_gru[['score']], data, topk=1)
    assert not report_df_gru.empty

    # Cleanup
    import shutil
    shutil.rmtree('data')
    print("Full Pipeline Test Passed.")

if __name__ == "__main__":
    test_full_pipeline()
