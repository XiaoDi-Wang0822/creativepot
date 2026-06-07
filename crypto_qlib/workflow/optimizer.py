import itertools
import yaml
import copy
from .manager import WorkflowManager

class GoalOptimizer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        self.goals = self.base_config.get('optimization_goals', {})
        self.search_space = self.base_config.get('search_space', {})

    def _generate_params(self):
        keys = self.search_space.keys()
        values = self.search_space.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def run(self, max_trials=20, rolling=True):
        print(f"Starting Goal-Driven Optimization (Rolling={rolling})...")
        trial, best_metrics, best_config = 0, None, None
        param_generator = self._generate_params()

        while trial < max_trials:
            try: params = next(param_generator)
            except StopIteration: break
            trial += 1
            print(f"\n--- Trial {trial}: {params} ---")

            current_config = copy.deepcopy(self.base_config)
            for k, v in params.items():
                if k in current_config['model'].get('params', {}): current_config['model']['params'][k] = v
                elif k in current_config['backtest']: current_config['backtest'][k] = v
                elif k in current_config['model']: current_config['model'][k] = v

            temp_config_path = 'configs/temp_opt_config.yaml'
            with open(temp_config_path, 'w') as f: yaml.dump(current_config, f)

            try:
                wm = WorkflowManager(temp_config_path)
                metrics = wm.run_experiment(rolling=rolling, silent=True)
                print(f"Metrics: Sharpe={metrics['Sharpe Ratio']:.2f}, PF={metrics['Profit Factor']:.2f}")

                met_goals = all(metrics.get(gk, 0) >= tv for gk, tv in self.goals.items())
                if met_goals:
                    print(f"!!! ALL GOALS MET in {trial} trials !!!")
                    return current_config, metrics

                if best_metrics is None or metrics['Sharpe Ratio'] > best_metrics['Sharpe Ratio']:
                    best_metrics, best_config = metrics, current_config
            except Exception as e:
                print(f"Trial failed: {e}")

        print("\nOptimization finished without meeting all goals.")
        return best_config, best_metrics
