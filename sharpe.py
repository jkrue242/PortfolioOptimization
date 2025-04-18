import torch.nn as nn

class SharpeLoss(nn.Module):
  def __init__(self):
    super().__init__()

    # to avoid div by 0
    self.eps = 1e-6

  def forward(self, weights, returns):
    # returns
    returns = (weights * returns).sum(dim=1)

    # mean returns
    mean_returns = returns.mean()

    # std of returns
    std_excess = returns.std(unbiased=False) + self.eps

    # sharpe ratio
    sharpe = mean_returns / std_excess

    # negative because we minimize loss
    return -sharpe