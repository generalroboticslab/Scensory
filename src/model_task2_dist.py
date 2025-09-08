import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SensorAttention(nn.Module):
    """
    Custom attention pooling for sensor readings
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
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

class DistanceRegressor(nn.Module):
    """
    Distance regression model for sensor data with enhanced unidirectional processing
    """
    def __init__(self,
                input_dim=9,          # Number of sensor readings per timestep
                hidden_dim=128,       # Hidden dimension size
                num_layers=3,         # Increased layers to compensate for lost bidirectionality
                dropout=0.3,          # Dropout rate
                use_bidirectional=False  # Control bidirectionality
                ):
        super().__init__()
        
        self.use_bidirectional = use_bidirectional
        
        # Enhanced LSTM with more capacity
        if use_bidirectional:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True, dropout=dropout)
            lstm_output_dim = hidden_dim * 2
        else:
            # Compensate for lost bidirectionality with larger hidden size
            self.lstm = nn.LSTM(input_dim, hidden_dim * 2, num_layers=num_layers,
                               batch_first=True, bidirectional=False, dropout=dropout)
            lstm_output_dim = hidden_dim * 2
        
        # Input preprocessing - helps capture more patterns
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # Positional Encoding with dropout
        self.pos_encoder = PositionalEncoding(lstm_output_dim)
        self.pos_dropout = nn.Dropout(dropout)
        
        # Enhanced MultiHead Attention with more heads
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=8,  # Increased from 4 to capture more patterns
            dropout=dropout
        )
        
        # Additional self-attention layer for better context modeling
        self.attention2 = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # SensorAttention pooling
        self.sensor_attention = SensorAttention(lstm_output_dim)
        
        # Enhanced shared representation with residual connection
        self.shared_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim * 2),
            nn.GroupNorm(1, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced regression head with skip connection
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Residual connection for pooled features
        self.residual_proj = nn.Linear(lstm_output_dim, hidden_dim)
        
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
        # x: (batch, seq_len, input_dim)
        
        # Enhanced input preprocessing
        x_proj = self.input_projection(x)
        
        # LSTM feature extraction
        lstm_out, _ = self.lstm(x_proj)  # (batch, seq_len, lstm_output_dim)
        
        # Add positional encoding
        x_pe = self.pos_dropout(self.pos_encoder(lstm_out))
        
        # First attention layer
        x_pe_perm = x_pe.permute(1, 0, 2)  # (seq_len, batch, features)
        attn_out1, _ = self.attention(x_pe_perm, x_pe_perm, x_pe_perm)
        
        # Second attention layer with residual connection
        attn_out2, _ = self.attention2(attn_out1, attn_out1, attn_out1)
        attn_out2 = attn_out2 + attn_out1  # Residual connection
        
        attn_out = attn_out2.permute(1, 0, 2)  # (batch, seq_len, features)
        
        # SensorAttention pooling
        pooled = self.sensor_attention(attn_out)
        
        # Shared layer processing
        shared = self.shared_layer(pooled)
        
        # Residual connection from pooled features
        residual = self.residual_proj(pooled)
        shared = shared + residual
        
        # Distance prediction
        distance = self.regressor(shared)
        
        return distance.squeeze(-1)
    
    def compute_loss(self, predictions, targets):
        """
        Compute MSE loss for distance regression
        
        Args:
            predictions: distance predictions from model
            targets: ground truth distance values
        """
        return F.mse_loss(predictions, targets)