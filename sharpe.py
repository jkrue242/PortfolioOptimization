import torch
import torch.nn as nn
import torch.nn.functional as F


class SharpeLoss(nn.Module):
    def __init__(self, epsilon=1e-6, risk_free_rate=0.0):
        super().__init__()
        self.epsilon = epsilon
        self.risk_free_rate = risk_free_rate

    def forward(self, weights, returns):
        portfolio_returns = torch.sum(weights * returns, dim=1)
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + self.epsilon

        sharpe_ratio = mean_return / std_return
        return -sharpe_ratio


def calculate_sharpe(weights, returns):
    portfolio_returns = torch.sum(weights * returns, dim=1)
    mean_return = torch.mean(portfolio_returns)
    std_return = torch.std(portfolio_returns)

    if std_return == 0:
        return 0.0

    return (mean_return / std_return).item()
