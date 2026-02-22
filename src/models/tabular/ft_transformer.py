import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, n_features, d_token))
        self.bias = nn.Parameter(torch.randn(1, n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            n_classes: int = 3,
            hidden_dims: list = [192],
            n_layers: int = 3,
            n_heads: int = 8,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.d_token = hidden_dims[0] if hidden_dims else 192
        self.tokenizer = FeatureTokenizer(input_dim, self.d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_token,
            nhead=n_heads,
            dim_feedforward=int(self.d_token * 1.33),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_token),
            nn.ReLU(),
            nn.Linear(self.d_token, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.tokenizer(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1)
        x_out = self.transformer(x_seq)
        cls_output = x_out[:, 0, :]
        return self.head(cls_output)
