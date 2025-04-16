import torch
import torch.nn as nn
import math
import numpy as np  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, params):
        super(TransformerModel, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(params['input_dim'], params['d_model'])
        self.pos_encoder = PositionalEncoding(params['d_model'])
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=params['d_model'],
            nhead=params['nhead'],
            dim_feedforward=params['d_model'] * 4,
            dropout=params['dropout']
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=params['num_layers'])
        
        # Output layer
        self.decoder = nn.Linear(params['d_model'], params['input_dim'])
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        
        # Embed the input
        src = self.embedding(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transformer expects (seq_len, batch_size, d_model)
        src = src.transpose(0, 1)
        
        # Pass through transformer
        output = self.transformer_encoder(src)
        
        # Transpose back to (batch_size, seq_len, d_model)
        output = output.transpose(0, 1)
        
        # Decode to original dimension
        output = self.decoder(output)
        
        return output

def train_model(model, X_train, y_train, X_val, y_val, optimizer, criterion, num_epochs=100):
    train_losses = []   
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs[:, -1, :], torch.FloatTensor(y_train))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs[:, -1, :], torch.FloatTensor(y_val))
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses