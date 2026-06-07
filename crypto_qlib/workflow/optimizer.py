import itertools
import random
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
        # Create combinations of hyperparameters
        keys = self.search_space.keys()
        values = self.search_space.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def run(self, max_trials=20):
        print("Starting Goal-Driven Optimization...")
        trial = 0
        best_metrics = None
        best_config = None

        param_generator = self._generate_params()

        while trial < max_trials:
            try:
                params = next(param_generator)
            except StopIteration:
                print("Search space exhausted.")
                break

            trial += 1
            print(f"\n--- Trial {trial}: Testing parameters {params} ---")

            # Update config with new params
            current_config = copy.deepcopy(self.base_config)
            for k, v in params.items():
                if k in current_config['model'].get('params', {}):
                    current_config['model']['params'][k] = v
                elif k in current_config['backtest']:
                    current_config['backtest'][k] = v
                elif k in current_config['model']:
                    current_config['model'][k] = v

            # Save temporary config
            temp_config_path = 'configs/temp_opt_config.yaml'
            with open(temp_config_path, 'w') as f:
                yaml.dump(current_config, f)

            # Run workflow
            try:
                wm = WorkflowManager(temp_config_path)
                # We assume data is already downloaded to save time
                metrics = wm.run_experiment_silent()

                print(f"Metrics: Sharpe={metrics['Sharpe Ratio']:.2f}, PF={metrics['Profit Factor']:.2f}")

                # Check goals
                met_goals = True
                for goal_key, target_val in self.goals.items():
                    if metrics.get(goal_key, 0) < target_val:
                        met_goals = False
                        break

                if met_goals:
                    print("!!! ALL GOALS MET !!!")
                    print(f"Achieved in {trial} trials.")
                    return current_config, metrics

                if best_metrics is None or metrics['Sharpe Ratio'] > best_metrics['Sharpe Ratio']:
                    best_metrics = metrics
                    best_config = current_config

            except Exception as e:
                print(f"Trial failed: {e}")

        print("\nOptimization finished without meeting all goals.")
        return best_config, best_metrics

# Add run_experiment_silent to WorkflowManager in manager.py
