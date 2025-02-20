import torchinfo
import torch
from torch import nn, Tensor
from liger_kernel.transformers import LigerRMSNorm as RMSNorm

"""

Inspired by the paper https://arxiv.org/html/2411.13766v1 which introduces the TinyAlign framework.

"""

class BridgeFormer(nn.Module):
    """This is the BridgeFormert component of the TinyAlign framework"""

    def _setup_mlp(self, input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            RMSNorm(hidden_dim),
            nn.GELU(),
        )

    def __init__(self, D_a: int, T: int, D_l: int, H: int = 256, M: int = 4, E: int = 4):
        super().__init__()
        
        self.input_mlp = self._setup_mlp(D_a, H)
        encoders = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=M,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoders, num_layers=E)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(output_size=T)
        self.output_mlp = self._setup_mlp(H, D_l)


        
    def forward(self, audio_features: Tensor) -> Tensor:
        # initial projection
        output_features = self.input_mlp(audio_features)
        # transformer encoder
        output_features = self.transformer_encoder(output_features)
        # adaptive pooling
        output_features = self.adaptive_pooling(output_features.permute(0, 2, 1)).permute(0, 2, 1)
        # final projection
        output_features = self.output_mlp(output_features)
        return output_features

if __name__ == "__main__":
    # Example usage
    audio_features = torch.randn(1, 1000, 80).to("cuda")
    model = BridgeFormer(D_a=80, T=100, D_l=80).to("cuda")
    model.to("cuda")
    torchinfo.summary(model)
    output = model(audio_features)