import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

class BaseModel:
    def fit(self, train_df, valid_df=None):
        raise NotImplementedError
    def predict(self, test_df):
        raise NotImplementedError

class LGBModel(BaseModel):
    def __init__(self, params=None):
        self.params = params or {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
        self.model = None
    def fit(self, train_df, valid_df=None):
        X_train = train_df.drop(columns=['label', 'instrument', 'datetime'], errors='ignore')
        y_train = train_df['label']
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets, valid_names = [train_data], ['train']
        if valid_df is not None:
            X_valid = valid_df.drop(columns=['label', 'instrument', 'datetime'], errors='ignore')
            y_valid = valid_df['label']
            valid_sets.append(lgb.Dataset(X_valid, label=y_valid, reference=train_data)); valid_names.append('valid')
        self.model = lgb.train(self.params, train_data, num_boost_round=1000, valid_sets=valid_sets, valid_names=valid_names,
                               callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)])
    def predict(self, test_df):
        return self.model.predict(test_df.drop(columns=['label', 'instrument', 'datetime'], errors='ignore'))

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(GRUNet, self).__init__()
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0); return self.fc(out[:, -1, :])

class TSDataSet(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len=10):
        self.X, self.y, self.seq_len = X.astype(np.float32), y.astype(np.float32), seq_len
    def __len__(self): return len(self.X) - self.seq_len + 1
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx : idx + self.seq_len]), torch.from_numpy(np.array([self.y[idx + self.seq_len - 1]]))

class GRUModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, lr=0.001, batch_size=1024, epochs=10, seq_len=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GRUNet(input_dim, hidden_dim, 1, num_layers).to(self.device)
        self.lr, self.batch_size, self.epochs, self.seq_len = lr, batch_size, epochs, seq_len
    def fit(self, train_df, valid_df=None):
        all_dataloaders = []
        for inst, group in train_df.groupby('instrument'):
            X, y = group.drop(columns=['label', 'instrument', 'datetime'], errors='ignore').values, group['label'].values
            if len(X) < self.seq_len: continue
            all_dataloaders.append(torch.utils.data.DataLoader(TSDataSet(X, y, self.seq_len), batch_size=self.batch_size, shuffle=True))
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr); criterion = nn.MSELoss(); self.model.train()
        for epoch in range(self.epochs):
            total_loss, count = 0, 0
            for dl in all_dataloaders:
                for bx, by in dl:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad(); loss = criterion(self.model(bx), by); loss.backward(); optimizer.step()
                    total_loss += loss.item(); count += 1
            if count > 0: print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/count:.6f}")
    def predict(self, test_df):
        self.model.eval(); preds_list = []
        for inst, group in test_df.groupby('instrument'):
            X = group.drop(columns=['label', 'instrument', 'datetime'], errors='ignore').values.astype(np.float32)
            X_p = np.concatenate([np.zeros((self.seq_len-1, X.shape[1]), dtype=np.float32), X], axis=0)
            with torch.no_grad():
                inst_preds = [self.model(torch.from_numpy(X_p[i:i+self.seq_len]).unsqueeze(0).to(self.device)).cpu().item() for i in range(len(X))]
            preds_list.append(pd.Series(inst_preds, index=group.index))
        return pd.concat(preds_list).sort_index()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model); position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerNet(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerNet, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model); self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers); self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_fc(x); x = self.pos_encoder(x); x = self.transformer_encoder(x); return self.fc(x[:, -1, :])

class TransformerModel(GRUModel):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, lr=0.001, batch_size=1024, epochs=10, seq_len=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerNet(input_dim, d_model, nhead, num_layers).to(self.device)
        self.lr, self.batch_size, self.epochs, self.seq_len = lr, batch_size, epochs, seq_len
