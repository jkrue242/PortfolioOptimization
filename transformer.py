import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=500): # Corrected from init to __init__
    super().__init__()

    # vector of positional encoding = (max_len, d_model)
    positional_encoding_vec = torch.zeros(max_len, d_model)

    # vector of positions = (max_len, 1)
    position_vec = torch.arange(0, max_len).unsqueeze(1).float()

    # divisor term for each dimension
    div = torch.exp(
      torch.arange(0, d_model, 2).float() *
      (-np.log(10000.0) / d_model)
    )

    # sinusoidal encoding for even dimension
    positional_encoding_vec[:, 0::2] = torch.sin(position_vec * div)

    # cosine encoding for odd dimention
    positional_encoding_vec[:, 1::2] = torch.cos(position_vec * div)

    # positional_encoding is constant and not a parameter so not
    # updated by gradient
    self.register_buffer('positional_encoding', positional_encoding_vec.unsqueeze(0))

  def forward(self, x):
    return x + self.positional_encoding[:, :x.size(1), :]
  


class PortfolioTransformer(nn.Module):
  def __init__(self, seq_len, num_features, num_assets,
                d_model=128, nhead=8, num_layers=2, dim_ff=256, dropout=0.1, min_weight=1e-6): # Added min_weight
    super().__init__()

    # input layer
    self.input_layer = nn.Linear(num_features, d_model)

    # position encoding layer
    self.pos_enc_layer = PositionalEncoding(d_model, max_len=seq_len)

    # transformer encoder layer
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_ff, dropout=dropout, activation='relu')

    # transormer encoder
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # avg pool layer
    self.avg_pool_layer = nn.AdaptiveAvgPool1d(1)

    # output layer
    self.output_layer = nn.Linear(d_model, num_assets)

    self.min_weight = min_weight
    self.num_assets = num_assets

  def forward(self, x):
    # input
    x = self.input_layer(x)

    # encode position
    x = self.pos_enc_layer(x)
    x = x.permute(1, 0, 2)

    # transformer
    x = self.transformer_encoder(x)
    x = x.permute(1, 2, 0)

    # avg pool
    x = self.avg_pool_layer(x).squeeze(2)

    # output
    output = self.output_layer(x)

    # softmax
    weights = F.softmax(output, dim=1)


    # if self.min_weight > 0:
    #   scaling_factor = 1.0 - self.num_assets * self.min_weight
    #   valid_weights = self.min_weight + scaling_factor * softmax_weights
    # else:
    #   valid_weights = softmax_weights

    return weights