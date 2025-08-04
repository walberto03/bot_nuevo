import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, input_size, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        # x: [batch, seq, features] → transpose para transformer
        x = x.permute(1, 0, 2)
        y = self.transformer(x)
        return self.fc(y[-1])
