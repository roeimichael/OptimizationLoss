import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.3, d_hidden_factor: int = 2):
        super().__init__()
        d_inner = d_model * d_hidden_factor
        self.block = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(d_inner),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            n_classes: int = 3,
            d_model: int = 256,
            n_layers: int = 4,
            dropout: float = 0.3,
            **kwargs
    ):
        super().__init__()

        if 'hidden_dims' in kwargs and kwargs['hidden_dims']:
            d_model = kwargs['hidden_dims'][0]
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(d_model, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        return self.head(x)
