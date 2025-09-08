import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0)]
        return x.transpose(0, 1)  # [batch, seq_len, d_model]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)

        self.net = nn.Sequential(
            self.conv1,
            nn.BatchNorm1d(n_outputs),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            nn.BatchNorm1d(n_outputs),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        if out.shape[-1] != res.shape[-1]:
            out = out[:, :, :res.shape[-1]]
        return self.relu(out + res)

class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, devices, seq_len, hidden_dim = x.size()
        x_reshaped = x.permute(2, 0, 1, 3).reshape(seq_len, batch * devices, hidden_dim)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped, attn_mask=mask)
        attn_out = self.dropout(attn_out)
        out = self.norm(x_reshaped + attn_out)
        out = out.view(seq_len, batch, devices, hidden_dim).permute(1, 2, 0, 3)
        return out

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, devices, hidden_dim = x.size()
        x_reshaped = x.permute(1, 0, 2, 3).reshape(seq_len, batch * devices, hidden_dim)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped, attn_mask=mask)
        attn_out = self.dropout(attn_out)
        out = self.norm(x_reshaped + attn_out)
        out = out.view(seq_len, batch, devices, hidden_dim).permute(1, 0, 2, 3)
        return out

class SpeciesClassifier(nn.Module):
    def __init__(self, input_channels: int = 9, num_classes: int = 5,
                 hidden_dim: int = 64, kernel_size: int = 5, dropout: float = 0.3):
        super().__init__()

        # Shared Temporal Convolutional Network (TCN)
        self.tcn1 = TemporalBlock(input_channels, hidden_dim, kernel_size, stride=1, dilation=1, dropout=dropout)
        self.tcn2 = TemporalBlock(hidden_dim, hidden_dim, kernel_size, stride=1, dilation=2, dropout=dropout)

        # Attention Modules
        self.spatial_attn = SpatialAttention(hidden_dim)
        self.temporal_attn = TemporalAttention(hidden_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Classification Head
        self.fc_class1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_class2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'fc' in name and 'weight' in name:
                nn.init.kaiming_normal_(param)

    def forward(self, x: torch.Tensor):
        batch_size, num_devices, seq_len, in_channels = x.size()

        # Pass each device's data through the same TCN
        device_outputs = []
        for i in range(num_devices):
            device_input = x[:, i].permute(0, 2, 1)  # [batch, in_channels, seq_len]
            device_output = self.tcn2(self.tcn1(device_input))  # [batch, hidden_dim, seq_len]
            device_outputs.append(device_output.permute(0, 2, 1))  # [batch, seq_len, hidden_dim]

        # Stack: [batch, devices, seq_len, hidden_dim]
        spatial_features = torch.stack(device_outputs, dim=1)

        # Spatial Attention
        spatial_features = self.spatial_attn(spatial_features)

        # Temporal Attention
        temporal_features = spatial_features.transpose(1, 2)  # [batch, seq_len, devices, hidden_dim]
        temporal_features = self.temporal_attn(temporal_features)

        # Average across devices
        x_shared = temporal_features.mean(dim=2)  # [batch, seq_len, hidden_dim]

        # Positional Encoding
        x_shared = self.pos_encoder(x_shared)  # [batch, seq_len, hidden_dim]

        # Global average pooling across time
        x_shared = x_shared.mean(dim=1)  # [batch, hidden_dim]

        # Classification Head
        class_x = F.relu(self.fc_class1(x_shared))
        class_x = self.dropout(class_x)
        class_logits = self.fc_class2(class_x)  # [batch, num_classes]

        return class_logits