import torch
from torch.utils.data import Dataset
import yfinance as yf
import pandas as pd


class PortfolioDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.num_samples = len(data) - window_size
        self.tickers = data.columns.get_level_values(0).unique()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        window_data = self.data.iloc[idx : idx + self.window_size]
        next_returns = self.data.iloc[idx + self.window_size]

        returns = []
        for ticker in self.tickers:
            returns.append(next_returns[ticker]["return"])

        features = torch.tensor(window_data.values, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        return {"features": features, "returns": returns}


def load_data(ETFS, START_DATE, END_DATE, ROLLING_AVGS):
    ticker_df = yf.download(
        ETFS, start=START_DATE, end=END_DATE, group_by="ticker", auto_adjust=True
    )
    features = []
    for e in ETFS:
        data = ticker_df[e].copy()

        data["return"] = data["Close"].pct_change().fillna(0)
        data["cumulative_return"] = (1 + data["return"]).cumprod() - 1

        for i in ROLLING_AVGS:
            data[f"ma_{i}"] = data["Close"].rolling(i).mean().fillna(method="bfill")

        features.append(
            data[["return"] + ["cumulative_return"] + [f"ma_{i}" for i in ROLLING_AVGS]]
        )

    data = pd.concat(features, axis=1, keys=ETFS).dropna()
    print(f"Number of features per asset: {len(data.columns) // len(ETFS)}")
    return data


def split_data(data, train_pct=0.8, val_pct=0.1):
    total_samples = len(data)
    train_size = int(train_pct * total_samples)
    val_size = int(val_pct * total_samples)

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size : train_size + val_size]
    test_data = data.iloc[train_size + val_size :]

    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")

    return train_data, val_data, test_data
