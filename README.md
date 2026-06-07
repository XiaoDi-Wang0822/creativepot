# Crypto Qlib

這是一個專門為加密貨幣設計的高性能量化研究生態系統，架構模仿了 Microsoft Qlib。

## 核心模塊

- `data`: 高性能二進制存儲系統，包含自動化數據管線（Data Pipeline）。
- `features`: 提供 Alpha158、Alpha360 及加密貨幣特色因子。
- `models`: 封裝了 LightGBM、GRU 以及先進的 **Transformer** 模型。
- `strategy`: 提供 Top-K 等量化選幣策略。
- `backtest`: 支持分鐘級回測，包含資產特定的滑點與手續費模型。
- `trade`: 提供 **Dry Run (模擬盤)** 交易橋接器。
- `analysis`: 自動生成互動式 HTML 報告，包含 Sharpe Ratio、Profit Factor 等關鍵指標。
- `workflow`: 使用 YAML 配置文件驅動整個實驗流程。支持 **目標驅動優化 (Goal-Driven Optimization)**。

## 快速開始

1. 安裝依賴：
   ```bash
   pip install ccxt pandas numpy pyyaml plotly scipy statsmodels torch lightgbm tqdm
   ```

2. 數據下載與實驗運行：
   在 `configs/advanced_workflow.yaml` 配置好參數後運行：
   ```python
   from crypto_qlib.workflow.manager import WorkflowManager
   wm = WorkflowManager('configs/advanced_workflow.yaml')
   wm.run_experiment()
   ```

3. 目標驅動優化 (自動尋參直到達標)：
   您可以設置目標（如夏普率 > 1.5），系統會自動在搜尋空間內迭代。
   ```python
   from crypto_qlib.workflow.optimizer import GoalOptimizer
   opt = GoalOptimizer('configs/advanced_workflow.yaml')
   best_config, best_metrics = opt.run(max_trials=50)
   ```

4. 模擬盤運行 (Dry Run)：
   ```python
   from crypto_qlib.trade.bridge import DryRunBridge
   bridge = DryRunBridge(exchange_id='mexc', symbols=['BTC/USDT'], model=your_model)
   bridge.step()
   ```

## 輸出
系統會生成一個包含累積報酬、最大回撤、IC 分佈等圖表的 HTML 報告（如 `advanced_report.html`）。
