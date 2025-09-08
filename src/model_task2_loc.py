import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SensorAttention(nn.Module):
    """
    Attention mechanism for sensor readings
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        weights = F.softmax(self.attention(x), dim=1)  # (batch, seq_len, 1)
        return torch.sum(weights * x, dim=1)  # (batch, hidden_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


class SpatialBinClassifier(nn.Module):
    """
    Enhanced model for spatial bin prediction with TCN-inspired architecture
    """
    def __init__(self, 
                 input_dim=9,        # Number of sensor readings per timestep
                 seq_length=30,      # Window size
                 hidden_dim=128,     # Hidden dimension size
                 num_layers=2,       # Number of LSTM layers
                 dropout=0.3,        # Dropout rate
                 num_spatial_bins=4  # 4 bins (z<=0 only)
                ):
        super().__init__()
        
        # Temporal processing (TCN-inspired)
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        
        # Positional Encoding with dropout
        self.pos_encoder = PositionalEncoding(hidden_dim * 2)
        self.pos_dropout = nn.Dropout(dropout)
        
        # Enhanced attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, 
            num_heads=4,
            dropout=dropout
        )
        
        # Classification head (aligned with semantic model)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GroupNorm(1, hidden_dim),  # Better than BN for variable lengths
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_spatial_bins)
        )
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights using Kaiming normal and biases to zero."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:  # Only initialize 2D+ weights
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:  # Initialize biases to zero
                nn.init.zeros_(param)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Temporal processing
        lstm_out, _ = self.lstm1(x)  # (batch, seq_len, hidden_dim*2)
        
        # Add positional encoding
        lstm_out = self.pos_dropout(self.pos_encoder(lstm_out))
        
        # Attention mechanism (full self-attention)
        lstm_out = lstm_out.permute(1, 0, 2)  # (seq_len, batch, features)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)  # (batch, seq_len, features)
        
        # Pooling and classification
        pooled = attn_out.mean(dim=1)  # Global average pooling
        return self.classifier(pooled)