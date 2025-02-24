import torchinfo
import torch
from torch import nn, Tensor
from liger_kernel.transformers import LigerRMSNorm as RMSNorm

"""

Inspired by the paper https://arxiv.org/html/2411.13766v1 which introduces the TinyAlign framework.

"""

class MLP(nn.Module):
    """Multi-layer perceptron with RMS normalization and GELU activation."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Group normalization layers
        self.norms = nn.ModuleList([
            RMSNorm(output_dim) for _ in range(3)
        ])
        
        # Group linear layers
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False),
            nn.Linear(output_dim, output_dim, bias=False),
            nn.Linear(output_dim, output_dim, bias=False)
        ])
        
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        for linear, norm in zip(self.linears, self.norms):
            x = linear(x)
            x = norm(x)
            x = self.act(x)
        return x


class BridgeFormer(nn.Module):
    """BridgeFormer component of the TinyAlign framework.
    Args:
        D_a (int): Input audio feature dimension size
        T (int): Target token length
        D_l (int): Output feature dimension (LLM embedding dimenison)
        H (int): Hidden dimension (default: 256)
        M (int): Number of attention heads (default: 4)
        E (int): Number of encoder layers (default: 4)
    """

    def __init__(self, D_a: int, T: int, D_l: int, H: int = 256, M: int = 4, E: int = 4):
        super().__init__()
        
        # Input projection
        self.input_mlp = MLP(D_a, H)
        
        # Transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=M,
            batch_first=True,
            norm_first=True  # Pre-norm is generally more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=E)
        
        # Pooling and output projection
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(output_size=T)
        self.output_mlp = MLP(H, D_l)
        
        self._dimensions = {'D_a': D_a, 'T': T, 'D_l': D_l, 'H': H}

    def forward(self, audio_features: Tensor) -> Tensor:
        # Input projection: [batch, seq_len, D_a] -> [batch, seq_len, H]
        features: Tensor = self.input_mlp(audio_features)
        
        # Transformer encoding: [batch, seq_len, H] -> [batch, seq_len, H]
        features: Tensor = self.transformer_encoder(features)
        
        # Adaptive pooling: [batch, seq_len, H] -> [batch, T, H]
        features: Tensor = features.permute(0, 2, 1)  # [batch, H, seq_len]
        features: Tensor = self.adaptive_pooling(features)  # [batch, H, T]
        features: Tensor = features.permute(0, 2, 1)  # [batch, T, H]
        
        # Output projection: [batch, T, H] -> [batch, T, D_l]
        features = self.output_mlp(features)
        
        return features

if __name__ == "__main__":
    # Example usage
    audio_features = torch.randn(1, 1000, 80).to("cuda")
    model = BridgeFormer(D_a=80, T=100, D_l=80).to("cuda")
    model.to("cuda")
    torchinfo.summary(model)
    output = model(audio_features)