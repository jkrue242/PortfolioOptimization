import torch 
from torch.utils.data import Dataset, DataLoader


class PortfolioDataset(Dataset):
  def __init__(self, df, lookback):
    # days to lookback
    self.lookback = lookback

    # np array shape (days, n_tickers, n_features)
    self.data = df.values

    # returns cross section, and shift by 1 day, and then convert to np array
    self.returns = df.xs('return', level=1, axis=1).shift(-1).fillna(0).values
    self.length = len(self.data) - (lookback + 1)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    # end of window
    offset = idx + self.lookback

    # feature shape = (lookback, total_features)
    x = self.data[idx : offset]

    # next day returns for every etf = (n_tickers,)
    y = self.returns[offset]
    return {
      'features': torch.tensor(x, dtype=torch.float),
      'returns':  torch.tensor(y, dtype=torch.float),
    }