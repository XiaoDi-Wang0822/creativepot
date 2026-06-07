import yaml
import os
import pandas as pd
from crypto_qlib.data.provider import DataProvider
from crypto_qlib.features.extractor import FeatureExtractor
from crypto_qlib.models.wrapper import LGBModel, GRUModel
from crypto_qlib.backtest.engine import SimpleBacktester
from crypto_qlib.analysis.analyzer import Analyzer

class WorkflowManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.provider = DataProvider(bin_dir=self.config['data']['bin_dir'])

    def run(self):
        # 1. Load Data
        print("Loading data...")
        data = self.provider.load_data(
            self.config['data']['instruments'],
            self.config['data']['start_time'],
            self.config['data']['end_time']
        )

        # 2. Extract Features
        print("Extracting features...")
        extractor = FeatureExtractor(data)
        features = extractor.add_alpha158().add_crypto_specific().add_labels().get_features()

        # 3. Split Train/Test
        train_end = self.config['data']['train_end']
        train_df = features[features.index.get_level_values('datetime') <= train_end]
        test_df = features[features.index.get_level_values('datetime') > train_end]

        # 4. Train Model
        print(f"Training {self.config['model']['type']}...")
        if self.config['model']['type'] == 'LightGBM':
            model = LGBModel(self.config['model'].get('params'))
        elif self.config['model']['type'] == 'GRU':
            model = GRUModel(input_dim=train_df.shape[1]-1)
        else:
            raise ValueError("Unknown model type")

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
            slippage_ticks=self.config['backtest']['slippage_ticks'],
            tick_size=self.config['backtest']['tick_size']
        )
        report_df = backtester.run(test_df[['score']], data, topk=self.config['backtest']['topk'])

        # 7. Analysis
        print("Analyzing...")
        ic, rank_ic = Analyzer.calculate_ic(test_df[['score']], test_df[['label']])
        metrics = Analyzer.get_backtest_metrics(report_df)
        print("Final Metrics:", metrics)

        Analyzer.generate_report(report_df, ic, rank_ic, self.config['analysis']['output_report'])
        print("Workflow Completed.")
