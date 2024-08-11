
import torch
import torch.nn as nn

# SimplePoseNet: A simple fully connected neural network for pose estimation
class SimplePoseNet(nn.Module):
    def __init__(self, num_joints, num_frames, input_dim, output_dim):
        super(SimplePoseNet, self).__init__()
        self.num_joints = num_joints
        self.output_dim = output_dim
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_frames * num_joints * input_dim, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_joints * output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(-1, 1, self.num_joints, self.output_dim)
        return x

# LSTM_PoseNet: An LSTM-based model for processing pose sequences
class LSTM_PoseNet(nn.Module):
    def __init__(self, num_joints, num_frames, input_dim, output_dim, hidden_dim=512, num_layers=1):
        super(LSTM_PoseNet, self).__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.output_dim = output_dim

        # LSTM layer that will process the entire sequence
        self.lstm = nn.LSTM(input_size=num_joints * input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        # Output layer that maps from hidden state space to the output space
        self.fc = nn.Linear(hidden_dim, num_joints * output_dim)

    def forward(self, x):
        # Reshape input to match LSTM input shape: (batch, seq_len, features)
        x = x.view(-1, self.num_frames, self.num_joints * self.input_dim)
        
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # We use the last hidden state to predict the output
        x = self.fc(lstm_out[:, -1, :])

        # Reshape to (batch_size, 1, num_joints, output_dim) for consistency with previous model's output shape
        x = x.view(-1, 1, self.num_joints, self.output_dim)
        return x

# PositionalEncoding: Provides positional encoding for transformer models
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        self.register_buffer('permanent_pe', self.pe)

    def forward(self, x):
        # Apply positional encoding to the second dimension (sequence dimension)
        return x + self.permanent_pe[:, :x.size(1), :]

# SimpleTransformerPoseNet: A transformer-based model for pose estimation with positional encoding
class SimpleTransformerPoseNet(nn.Module):
    def __init__(self, num_joints, num_frames, input_dim, output_dim, d_model=256, nhead=8, num_encoder_layers=3):
        super(SimpleTransformerPoseNet, self).__init__()
        self.num_joints = num_joints
        self.output_dim = output_dim
        self.input_linear = nn.Linear(num_joints * input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_frames)
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_encoder_layers)
        self.output_linear = nn.Linear(d_model, num_joints * output_dim)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # Reshape input to (batch, seq_len, num_joints * input_dim)
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Aggregate over sequence and project to output dimension
        x = self.output_linear(x)
        x = x.view(x.size(0), 1, self.num_joints, self.output_dim)  # Reshape to desired output format
        return x
