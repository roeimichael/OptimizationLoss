import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return x.squeeze(1)


class NeuralNetClassifierEnhanced(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[256, 128, 64],
        n_classes=3,
        dropout=0.3,
        use_residual=True,
        use_attention=False,
        activation='gelu'
    ):
        super().__init__()

        self.use_residual = use_residual
        self.use_attention = use_attention

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        else:
            self.activation = nn.ReLU()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            self.activation,
            nn.Dropout(dropout)
        )

        self.layers = nn.ModuleList()
        prev_dim = hidden_dims[0]

        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            if use_residual and prev_dim == hidden_dim:
                self.layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    self.activation,
                    nn.Dropout(dropout)
                ))
            prev_dim = hidden_dim

        if use_attention and len(hidden_dims) > 0:
            self.attention = SelfAttention(prev_dim, num_heads=4, dropout=dropout*0.5)

        self.output_layer = nn.Linear(prev_dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_projection(x)

        for layer in self.layers:
            x = layer(x)

        if self.use_attention:
            x = self.attention(x)

        return self.output_layer(x)


class NeuralNetClassifierMixup(NeuralNetClassifierEnhanced):
    def __init__(self, *args, mixup_alpha=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixup_alpha = mixup_alpha

    def mixup_data(self, x, y):
        if self.mixup_alpha > 0 and self.training:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)

            mixed_x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
        return x, y, y, 1.0


class EnsembleClassifier(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)


class NeuralNetClassifierWithUncertainty(NeuralNetClassifierEnhanced):
    def __init__(self, *args, num_samples=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

    def forward(self, x, return_uncertainty=False):
        if not return_uncertainty or not self.training:
            return super().forward(x)

        outputs = []
        for _ in range(self.num_samples):
            outputs.append(super().forward(x))

        outputs = torch.stack(outputs)
        mean_output = outputs.mean(dim=0)
        uncertainty = outputs.std(dim=0).mean(dim=-1)

        return mean_output, uncertainty
