import yaml
import os
import pandas as pd
from crypto_qlib.data.provider import DataProvider
from crypto_qlib.data.pipeline import DataPipeline
from crypto_qlib.features.extractor import FeatureExtractor
from crypto_qlib.models.wrapper import LGBModel, GRUModel, TransformerModel
from crypto_qlib.backtest.engine import SimpleBacktester
from crypto_qlib.analysis.analyzer import Analyzer

class WorkflowManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.provider = DataProvider(bin_dir=self.config['data']['bin_dir'])

    def run_data_pipeline(self):
        pipeline = DataPipeline(
            bin_dir=self.config['data']['bin_dir'],
            raw_dir=self.config['data']['raw_dir'],
            exchange_id=self.config['data']['exchange_id']
        )
        pipeline.run(
            limit=self.config['data']['limit'],
            timeframe=self.config['data']['timeframe'],
            start_time=self.config['data']['start_time'],
            end_time=self.config['data']['end_time']
        )

    def run_experiment(self):
        # 1. Load Data
        print("Loading data...")
        instruments = self.provider.get_instruments()
        if not instruments:
            print("No data found. Running data pipeline first...")
            self.run_data_pipeline()
            instruments = self.provider.get_instruments()

        data = self.provider.load_data(
            instruments,
            self.config['data']['start_time'],
            self.config['data']['end_time']
        )

        # 2. Extract Features
        print("Extracting features...")
        extractor = FeatureExtractor(data)
        features = extractor.add_alpha158().add_crypto_specific().add_labels().get_features().dropna()

        # 3. Split Train/Test
        train_end = pd.to_datetime(self.config['data']['train_end']).replace(tzinfo=None)
        features_dt = features.index.get_level_values('datetime')
        train_df = features[features_dt <= train_end]
        test_df = features[features_dt > train_end]

        # 4. Train Model
        print(f"Training {self.config['model']['type']}...")
        input_dim = train_df.shape[1] - 1 # excluding label
        m_type = self.config['model']['type']
        m_params = self.config['model'].get('params', {})

        if m_type == 'LightGBM':
            model = LGBModel(m_params)
        elif m_type == 'GRU':
            model = GRUModel(input_dim=input_dim, **m_params)
        elif m_type == 'Transformer':
            model = TransformerModel(input_dim=input_dim, **m_params)
        else:
            raise ValueError(f"Unknown model type: {m_type}")

        model.fit(train_df)

        # 5. Predict
        print("Predicting...")
        preds = model.predict(test_df)
        test_df = test_df.copy()
        test_df['score'] = preds

        # 6. Backtest
        print("Backtesting...")
        backtester = SimpleBacktester(
            initial_cash=self.config['backtest']['cash'],
            commission=self.config['backtest']['commission'],
            slippage_ticks=self.config['backtest']['slippage_ticks']
        )
        report_df = backtester.run(test_df[['score']], data, topk=self.config['backtest']['topk'])

        # 7. Analysis
        print("Analyzing...")
        ic, rank_ic = Analyzer.calculate_ic(test_df[['score']], test_df[['label']])
        metrics = Analyzer.get_backtest_metrics(report_df)
        print("Final Metrics:", metrics)

        Analyzer.generate_report(report_df, ic, rank_ic, self.config['analysis']['output_report'])
        print(f"Workflow Completed. Report saved to {self.config['analysis']['output_report']}")
