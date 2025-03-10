import torchinfo
import torch
from torch import nn, Tensor
from liger_kernel.transformers import LigerRMSNorm as RMSNorm

"""

Inspired by the paper https://arxiv.org/html/2411.13766v1 which introduces the TinyAlign framework.

"""

class MLP(nn.Module):
    """Multi-layer perceptron with RMS normalization and GELU activation."""
    def __init__(self, input_dim: int, output_dim: int, act, hidden: int = 256):
        super().__init__()
        # Group normalization layers
        self.norms = nn.ModuleList([
            RMSNorm(hidden),
            RMSNorm(output_dim),
        ])
        
        # Group linear layers
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, hidden, bias=False),
            nn.Linear(hidden, output_dim, bias=False)
        ])

        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        for linear, norm in zip(self.linears, self.norms):
            x = linear(x)
            x = norm(x)
            x = self.act(x)
        return x


class BridgeFormer(nn.Module):
    """BridgeFormer component of the TinyAlign framework.
    Args:
        encoder_dim (int): Input audio feature dimension size
        token_length (int): Target token length
        embed_dim (int): Output feature dimension (LLM embedding dimenison)
        hidden_dim (int): Hidden dimension (default: 256)
        attention_heads (int): Number of attention heads (default: 4)
        encoder_layers (int): Number of encoder layers (default: 4)
    """

    def __init__(
            self,
            encoder_dim: int,
            token_length: int,
            embed_dim: int,
            hidden_dim: int = 256,
            attention_heads: int = 4,
            encoder_layers: int = 4
        ):
        super().__init__()

        # GeLU
        self.GELU = nn.GELU()

        # Input projection
        self.input_mlp = MLP(encoder_dim, hidden_dim, act=self.GELU)
        
        # Transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            batch_first=True,
            norm_first=True  # Pre-norm is generally more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        
        # Pooling and output projection
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(output_size=token_length)

        self.output_mlp = MLP(hidden_dim, embed_dim, act=self.GELU)
        
        self._dimensions = {'D_a': encoder_dim, 'T': token_length, 'D_l': embed_dim, 'H': hidden_dim}

    def forward(self, audio_features: Tensor) -> Tensor:
        # Input projection: [batch, seq_len, D_a] -> [batch, seq_len, H]
        features: Tensor = self.input_mlp(audio_features)
        
        # Transformer encoding: [batch, seq_len, H] -> [batch, seq_len, H]
        features: Tensor = self.transformer_encoder(features)
        
        # Conv1D, GeLU, Adaptive pooling: [batch, seq_len, H] -> [batch, T, H]
        features: Tensor = features.permute(0, 2, 1)  # [batch, H, seq_len]
        features: Tensor = self.conv(features)
        features: Tensor = self.GELU(features)
        features: Tensor = self.adaptive_pooling(features)  # [batch, H, T]
        features: Tensor = features.permute(0, 2, 1)  # [batch, T, H]
        
        # Output projection: [batch, T, H] -> [batch, T, D_l]
        features = self.output_mlp(features)

        return features

if __name__ == "__main__":
    # Example usage
    audio_features = torch.randn(1, 1000, 80).to("cuda")
    model = BridgeFormer(encoder_dim=80, token_length=100, embed_dim=80).to("cuda")
    model.to("cuda")
    torchinfo.summary(model)
    output = model(audio_features)