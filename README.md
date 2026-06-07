# Crypto Qlib (Ultimate Version)

這是一個專門為加密貨幣設計的高性能量化研究生態系統，完整複製了 Microsoft Qlib 的核心架構，並針對加密市場優化。

## 核心亮點

- **高性能二進制數據層 (High-Performance Data Layer)**：仿照 Qlib 實現 Column-based Binary Storage，秒級加載分鐘數據。
- **滾動式訓練與學習 (Rolling Training & Tasks)**：內置滾動窗口管理器，模擬真實交易中的定期重訓邏輯（Train -> Predict -> Shift）。
- **目標驅動優化 (Goal-Driven Optimization)**：自動尋優直至達成預設目標（如 Sharpe > 1.5, Profit Factor > 1.5）。
- **多模型支持 (AI Model Zoo)**：包含 LightGBM, GRU, 以及進階的 **Transformer**。
- **高精度回測 (Precision Backtest)**：支持 **3-Tick 資產特定滑點**、手續費模型與分鐘級選幣。

## 快速開始

1. 安裝依賴：
   ```bash
   pip install ccxt pandas numpy pyyaml plotly scipy statsmodels torch lightgbm tqdm
   ```

2. 運行滾動式實驗 (Rolling Experiment)：
   在 `configs/advanced_workflow.yaml` 配置 `rolling` 參數後運行：
   ```python
   from crypto_qlib.workflow.manager import WorkflowManager
   wm = WorkflowManager('configs/advanced_workflow.yaml')
   wm.run_experiment(rolling=True)
   ```

3. 自動尋優 (Optimization Loop)：
   ```python
   from crypto_qlib.workflow.optimizer import GoalOptimizer
   opt = GoalOptimizer('configs/advanced_workflow.yaml')
   best_config, best_metrics = opt.run(max_trials=50, rolling=True)
   ```

4. 模擬盤 (Dry Run)：
   ```python
   from crypto_qlib.trade.bridge import DryRunBridge
   bridge = DryRunBridge(exchange_id='mexc', symbols=['BTC/USDT', 'ETH/USDT'], model=your_model)
   bridge.step()
   ```

## 配置說明 (`advanced_workflow.yaml`)
- `rolling`: 設置 `train_len` (訓練長度) 和 `step_len` (滾動步長)。
- `optimization_goals`: 設置達標門檻。
- `search_space`: 設置超參數搜尋範圍。

## 輸出
系統會生成互動式 HTML 報告，包含累積報酬、最大回撤、IC 分佈及滾動訓練的各項性能指標。
