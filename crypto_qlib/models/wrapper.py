import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

class BaseModel:
    def fit(self, train_df, valid_df=None):
        raise NotImplementedError

    def predict(self, test_df):
        raise NotImplementedError

class LGBModel(BaseModel):
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
        }
        self.model = None

    def fit(self, train_df, valid_df=None):
        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']

        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ['train']

        if valid_df is not None:
            X_valid = valid_df.drop(columns=['label'])
            y_valid = valid_df['label']
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )

    def predict(self, test_df):
        X_test = test_df.drop(columns=['label'], errors='ignore')
        return self.model.predict(X_test)

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class TSDataSet(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len=10):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx : idx + self.seq_len]),
            torch.FloatTensor([self.y[idx + self.seq_len - 1]])
        )

class GRUModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, lr=0.001, batch_size=1024, epochs=10, seq_len=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GRUNet(input_dim, hidden_dim, 1, num_layers).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.seq_len = seq_len

    def fit(self, train_df, valid_df=None):
        # We need to handle MultiIndex: group by instrument to avoid cross-instrument sequences
        all_dataloaders = []
        for inst, group in train_df.groupby('instrument'):
            X = group.drop(columns=['label']).values
            y = group['label'].values
            if len(X) < self.seq_len:
                continue
            dataset = TSDataSet(X, y, self.seq_len)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            all_dataloaders.append(dataloader)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            count = 0
            for dataloader in all_dataloaders:
                for batch_x, batch_y in dataloader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1
            if count > 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/count:.6f}")

    def predict(self, test_df):
        self.model.eval()
        preds_list = []

        # Predict per instrument to maintain sequence integrity
        for inst, group in test_df.groupby('instrument'):
            X = group.drop(columns=['label'], errors='ignore').values
            # Padding for the beginning of the sequence where we don't have enough history
            # In a real system, we'd use data from the end of training
            padding = np.zeros((self.seq_len - 1, X.shape[1]))
            X_padded = np.concatenate([padding, X], axis=0)

            inst_preds = []
            with torch.no_grad():
                for i in range(len(X)):
                    seq = torch.FloatTensor(X_padded[i : i + self.seq_len]).unsqueeze(0).to(self.device)
                    pred = self.model(seq)
                    inst_preds.append(pred.cpu().item())

            preds_series = pd.Series(inst_preds, index=group.index)
            preds_list.append(preds_series)

        return pd.concat(preds_list).sort_index()
