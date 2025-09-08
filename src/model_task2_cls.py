import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(0.1)  # Added dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)  # Apply dropout

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)

        # Use GroupNorm instead of LayerNorm for correct dimension handling
        self.net = nn.Sequential(
            self.conv1,
            nn.GroupNorm(num_groups=1, num_channels=n_outputs),  # Changed
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            nn.GroupNorm(num_groups=1, num_channels=n_outputs),  # Changed
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.shape[-1] != res.shape[-1]:
            out = out[:, :, :res.shape[-1]]
        return self.relu(out + res)

class SingleSensorModel(nn.Module):
    def __init__(self,
                 input_channels: int = 9,
                 num_classes: int = 5,
                 hidden_dim: int = 128,
                 kernel_size: int = 5,
                 dropout: float = 0.3):
        super().__init__()

        # Temporal Convolutional Network (TCN)
        self.tcn1 = TemporalBlock(input_channels, hidden_dim, kernel_size, stride=1, dilation=1, dropout=dropout)
        self.tcn2 = TemporalBlock(hidden_dim, hidden_dim, kernel_size, stride=1, dilation=2, dropout=dropout)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)  # Added

        # Classification Head
        self.fc_class1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_class2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'fc' in name and 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'attention' in name:  # Added attention initialization
                if 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TCN with cross-block residual
        x = x.permute(0, 2, 1)
        out1 = self.tcn1(x)
        out2 = self.tcn2(out1)
        x = out1 + out2  # Residual connection between blocks
        x = x.permute(0, 2, 1)

        # Positional Encoding
        x = self.pos_encoder(x)

        # Multi-Head Attention
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, features]

        # Classification
        x = x.mean(dim=1)
        x = F.relu(self.fc_class1(x))
        x = self.dropout(x)
        return self.fc_class2(x)