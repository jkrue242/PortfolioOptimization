import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utils import PortfolioDataset
from sharpe import calculate_sharpe, SharpeLoss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        positional_encoding_vec = torch.zeros(max_len, d_model)
        position_vec = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        positional_encoding_vec[:, 0::2] = torch.sin(position_vec * div)
        positional_encoding_vec[:, 1::2] = torch.cos(position_vec * div)

        self.register_buffer(
            "positional_encoding", positional_encoding_vec.unsqueeze(0)
        )

    def forward(self, x):
        return x + self.positional_encoding[:, : x.size(1), :]


class PortfolioTransformer(nn.Module):
    def __init__(
        self,
        seq_len,
        num_features,
        num_assets,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_ff=128,
        dropout=0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.num_features = num_features
        self.num_assets = num_assets
        self.embedding = nn.Linear(num_features, d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.fc_out = nn.Linear(d_model, num_assets)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        weights = F.softmax(x, dim=-1)

        return weights


def train_model(
    train_data, val_data, etfs, sample_days, batch_size=64, epochs=200, device="cpu"
):
    train_dataset = PortfolioDataset(train_data, sample_days)
    val_dataset = PortfolioDataset(val_data, sample_days)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_features = len(train_data.columns) // len(etfs)

    model = PortfolioTransformer(
        seq_len=sample_days,
        num_features=len(etfs) * n_features,
        num_assets=len(etfs),
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_ff=128,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = SharpeLoss()

    history = {"train_losses": [], "val_losses": [], "val_sharpes": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            X = batch["features"].to(device)
            R = batch["returns"].to(device)

            optimizer.zero_grad()

            w = model(X)

            loss = loss_fn(w, R)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx == 0:
                print(
                    f"  Epoch {epoch:02d}, Batch {batch_idx:03d} - Predicted Weights (first sample):"
                )
                weights_to_print = w[0].detach().cpu().numpy()
                for i, etf in enumerate(etfs):
                    print(f"    {etf}: {weights_to_print[i]:.4f}", end=" ")
                print()

            train_loss = train_loss / len(train_loader)
            history["train_losses"].append(train_loss)

            model.eval()
            val_loss = 0
            all_weights = []
            all_returns = []

            with torch.no_grad():
                for batch in val_loader:
                    X = batch["features"].to(device)
                    R = batch["returns"].to(device)

                    w = model(X)

                    loss = loss_fn(w, R)
                    val_loss += loss.item()

                    all_weights.append(w)
                    all_returns.append(R)

            val_loss = val_loss / len(val_loader)
            history["val_losses"].append(val_loss)

            if all_weights and all_returns:
                val_weights = torch.cat(all_weights, dim=0)
                val_returns = torch.cat(all_returns, dim=0)
                val_sharpe = calculate_sharpe(val_weights, val_returns)
                history["val_sharpes"].append(val_sharpe)
            else:
                val_sharpe = 0
                history["val_sharpes"].append(val_sharpe)

            print(
                f"Epoch {epoch:02d}, Train Loss: {-train_loss:.4f}, Val Loss: {-val_loss:.4f}, Val Sharpe: {val_sharpe:.4f}"
            )

    return model, history


def evaluate_model(model, test_data, etfs, sample_days, device="cpu"):
    test_dataset = PortfolioDataset(test_data, sample_days)

    model.eval()
    predicted_weights = []
    actual_returns = []
    portfolio_returns = []
    dates = []
    test_dates = test_data.index[sample_days:]

    with torch.no_grad():
        for i in range(len(test_dataset)):
            batch = test_dataset[i]
            X = batch["features"].unsqueeze(0).to(device)
            R_actual = batch["returns"].cpu().numpy()

            weights = model(X).squeeze(0).cpu().numpy()
            predicted_weights.append(weights)
            actual_returns.append(R_actual)
            dates.append(test_dates[i] if i < len(test_dates) else None)

            Rp_t = np.sum(weights * R_actual)
            portfolio_returns.append(Rp_t)

    predicted_weights = np.array(predicted_weights)
    actual_returns = np.array(actual_returns)
    portfolio_returns = np.array(portfolio_returns)
    compounded_returns = (1 + portfolio_returns).cumprod()

    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    test_sharpe = mean_return / std_return if std_return > 0 else 0
    print(f"Test Set Sharpe Ratio: {test_sharpe:.4f}")

    cumulative_sharpe_ratios = []
    for j in range(len(portfolio_returns)):
        returns_subset = portfolio_returns[: j + 1]

        if len(returns_subset) > 1:
            mean_return = np.mean(returns_subset)
            std_return = np.std(returns_subset, ddof=0)

            if std_return != 0:
                sharpe = mean_return / std_return
            else:
                sharpe = 0
            cumulative_sharpe_ratios.append(sharpe)
        else:
            cumulative_sharpe_ratios.append(0)

    cumulative_sharpe_ratios = np.array(cumulative_sharpe_ratios)
    dates = [d for d in dates if d is not None]

    return {
        "weights": predicted_weights,
        "returns": actual_returns,
        "portfolio_returns": portfolio_returns,
        "compounded_returns": compounded_returns,
        "sharpe_ratio": test_sharpe,
        "cumulative_sharpe_ratios": cumulative_sharpe_ratios,
        "dates": dates,
    }
