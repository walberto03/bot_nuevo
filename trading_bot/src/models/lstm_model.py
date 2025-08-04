import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        # Dimensión de salida según bidireccionalidad
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        # Capa densa intermedia
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.relu = nn.ReLU()
        # Capa de salida
        self.fc2 = nn.Linear(lstm_output_size // 2, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)  # out: [batch, seq_len, hidden_size * (1 or 2)]
        last_step = out[:, -1, :]  # tomar última salida temporal
        x = self.fc1(last_step)
        x = self.relu(x)
        return self.fc2(x)  # retorna logits
