"""Feature Tokenizer Transformer for tabular data.

Implementation of FT-Transformer from "Revisiting Deep Learning Models for Tabular Data"
(Gorishniy et al., 2021). Uses feature tokenization and transformer architecture.
"""

from typing import List
import torch
import torch.nn as nn
import math


class FeatureTokenizer(nn.Module):
    """Tokenizes each feature into an embedding."""

    def __init__(self, input_dim: int, d_token: int):
        """Initialize feature tokenizer.

        Args:
            input_dim: Number of input features
            d_token: Dimension of each token embedding
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token

        # Linear projection for each feature
        self.feature_embeddings = nn.Linear(input_dim, input_dim * d_token)

        # Learnable bias for each token
        self.bias = nn.Parameter(torch.zeros(input_dim, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize features.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Tokens [batch_size, input_dim, d_token]
        """
        batch_size = x.size(0)

        # Project each feature to d_token dimensions
        # x: [batch_size, input_dim]
        # We want: [batch_size, input_dim, d_token]

        # Create feature-wise embeddings
        tokens = []
        for i in range(self.input_dim):
            # Extract single feature: [batch_size, 1]
            feature = x[:, i:i+1]
            # Project to d_token: [batch_size, d_token]
            token = feature * self.bias[i].unsqueeze(0)
            tokens.append(token)

        # Stack: [batch_size, input_dim, d_token]
        tokens = torch.stack(tokens, dim=1)

        return tokens


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            x: Input [batch_size, seq_len, d_model]

        Returns:
            Output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        context = torch.matmul(attn, V)  # [batch_size, n_heads, seq_len, d_k]

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.W_o(context)

        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input [batch_size, seq_len, d_model]

        Returns:
            Output [batch_size, seq_len, d_model]
        """
        # Multi-head attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data.

    Architecture:
    1. Feature Tokenization: Each feature -> token embedding
    2. CLS token prepended
    3. Transformer encoder layers
    4. Classification from CLS token
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [192],  # d_token dimension
        n_classes: int = 3,
        dropout: float = 0.3,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff_multiplier: int = 4
    ):
        """Initialize FT-Transformer.

        Args:
            input_dim: Number of input features
            hidden_dims: [d_token] - token embedding dimension
            n_classes: Number of output classes
            dropout: Dropout probability
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff_multiplier: Feed-forward dimension multiplier
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_token = hidden_dims[0] if hidden_dims else 192
        self.n_heads = n_heads

        # Feature tokenization
        self.tokenizer = FeatureTokenizer(input_dim, self.d_token)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_token,
                n_heads=n_heads,
                d_ff=self.d_token * d_ff_multiplier,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_token, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits [batch_size, n_classes]
        """
        batch_size = x.size(0)

        # Tokenize features: [batch_size, input_dim, d_token]
        tokens = self.tokenizer(x)

        # Add CLS token: [batch_size, 1 + input_dim, d_token]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Extract CLS token for classification
        cls_output = tokens[:, 0, :]  # [batch_size, d_token]

        # Classification head
        logits = self.head(cls_output)  # [batch_size, n_classes]

        return logits
