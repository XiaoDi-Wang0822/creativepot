# Crypto Qlib

這是一個專門為加密貨幣設計的高性能量化研究生態系統，架構模仿了 Microsoft Qlib。

## 核心模塊

- `data`: 高性能二進制存儲系統，支持分鐘級數據。
- `features`: 提供 Alpha158、Alpha360 及加密貨幣特色因子。
- `models`: 封裝了 LightGBM 與 GRU (PyTorch) 等模型。
- `strategy`: 提供 Top-K 等量化選幣策略。
- `backtest`: 支持分鐘級回測，包含滑點與手續費模型。
- `analysis`: 自動生成與 Qlib 風格一致的互動式 HTML 報告。
- `workflow`: 支持使用 YAML 配置文件驅動整個實驗流程。

## 快速開始

1. 安裝依賴：
   ```bash
   pip install ccxt pandas numpy pyyaml plotly scipy statsmodels torch lightgbm tqdm
   ```

2. 配置文件：
   參考 `configs/workflow_config.yaml` 設置數據路徑、模型參數與回測選項。

3. 運行實驗：
   ```python
   from crypto_qlib.workflow.manager import WorkflowManager
   wm = WorkflowManager('configs/workflow_config.yaml')
   wm.run()
   ```

## 輸出
系統會生成一個包含累積報酬、最大回撤、IC 分佈等圖表的 HTML 報告。
