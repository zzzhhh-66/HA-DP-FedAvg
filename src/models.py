import torch.nn as nn


class CreditRiskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)
