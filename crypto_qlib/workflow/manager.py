import yaml
import os
import pandas as pd
import numpy as np
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

    def _get_model(self, input_dim):
        m_type = self.config['model']['type']
        m_params = self.config['model'].get('params', {})
        if m_type == 'LightGBM': return LGBModel(m_params)
        if m_type == 'GRU': return GRUModel(input_dim=input_dim, **m_params)
        if m_type == 'Transformer': return TransformerModel(input_dim=input_dim, **m_params)
        raise ValueError(f"Unknown model type: {m_type}")

    def run_experiment(self, rolling=False, silent=False):
        if not silent: print("Loading data...")
        instruments = self.provider.get_instruments()
        if not instruments:
            if not silent: print("No data found. Running data pipeline...")
            self.run_data_pipeline()
            instruments = self.provider.get_instruments()

        data = self.provider.load_data(instruments, self.config['data']['start_time'], self.config['data']['end_time'])
        if not silent: print("Extracting features...")
        extractor = FeatureExtractor(data)
        features = extractor.add_alpha158().add_crypto_specific().add_labels().get_features().dropna()

        if not rolling:
            return self._run_single_task(features, data, silent)
        else:
            return self._run_rolling_tasks(features, data, silent)

    def _run_single_task(self, features, data, silent):
        train_end = pd.to_datetime(self.config['data']['train_end']).replace(tzinfo=None)
        features_dt = features.index.get_level_values('datetime')
        train_df = features[features_dt <= train_end]
        test_df = features[features_dt > train_end]

        if not silent: print(f"Training {self.config['model']['type']} (Single Task)...")
        model = self._get_model(train_df.shape[1]-1)
        model.fit(train_df)
        preds = model.predict(test_df)
        test_df = test_df.copy(); test_df['score'] = preds

        return self._evaluate(test_df, data, silent)

    def _run_rolling_tasks(self, features, data, silent):
        """
        Qlib-like Rolling Tasks
        """
        rolling_cfg = self.config['rolling']
        step_len = pd.Timedelta(rolling_cfg['step_len'])
        train_len = pd.Timedelta(rolling_cfg['train_len'])

        start_time = features.index.get_level_values('datetime').min()
        end_time = features.index.get_level_values('datetime').max()

        current_test_start = start_time + train_len
        all_preds = []

        while current_test_start + step_len <= end_time:
            train_start = current_test_start - train_len
            train_end = current_test_start
            test_end = current_test_start + step_len

            if not silent: print(f"Rolling Task: Train [{train_start} to {train_end}], Test [{train_end} to {test_end}]")

            f_dt = features.index.get_level_values('datetime')
            train_df = features[(f_dt >= train_start) & (f_dt < train_end)]
            test_df = features[(f_dt >= train_end) & (f_dt < test_end)]

            if len(train_df) > 0 and len(test_df) > 0:
                model = self._get_model(train_df.shape[1]-1)
                model.fit(train_df)
                preds = model.predict(test_df)
                test_df = test_df.copy(); test_df['score'] = preds
                all_preds.append(test_df[['score']])

            current_test_start += step_len

        if not all_preds:
            raise ValueError("No rolling windows were executed. Check train_len and step_len.")

        merged_test_df = pd.concat(all_preds).sort_index()
        # Join with labels for evaluation
        merged_test_df = merged_test_df.join(features[['label']], how='inner')

        return self._evaluate(merged_test_df, data, silent)

    def _evaluate(self, test_df, data, silent):
        if not silent: print("Backtesting & Analyzing...")
        backtester = SimpleBacktester(
            initial_cash=self.config['backtest']['cash'],
            commission=self.config['backtest']['commission'],
            slippage_ticks=self.config['backtest']['slippage_ticks']
        )
        report_df = backtester.run(test_df[['score']], data, topk=self.config['backtest']['topk'])
        ic, rank_ic = Analyzer.calculate_ic(test_df[['score']], test_df[['label']])
        metrics = Analyzer.get_backtest_metrics(report_df)

        if not silent:
            print("Final Metrics:", metrics)
            Analyzer.generate_report(report_df, ic, rank_ic, self.config['analysis']['output_report'])
        return metrics
