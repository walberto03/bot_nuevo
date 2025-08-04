import torch.nn as nn
import torch

class NewsSubmodel(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # Salida binaria (sube o baja)

    def forward(self, x):
        """
        x: tensor [batch_size, embedding_dim]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
