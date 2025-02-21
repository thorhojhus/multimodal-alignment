import torchinfo
import torch
from torch import nn, Tensor
from liger_kernel.transformers import LigerRMSNorm as RMSNorm

"""

Inspired by the paper https://arxiv.org/html/2411.13766v1 which introduces the TinyAlign framework.

"""

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.act = nn.GELU()
        self.norm1 = RMSNorm(output_dim)
        self.norm2 = RMSNorm(output_dim)
        self.norm3 = RMSNorm(output_dim)
        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.linear3 = nn.Linear(output_dim, output_dim, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.linear3(x)
        x = self.norm3(x)
        x = self.act(x)
        return x


class BridgeFormer(nn.Module):
    """This is the BridgeFormert component of the TinyAlign framework"""

    def __init__(self, D_a: int, T: int, D_l: int, H: int = 512, M: int = 4, E: int = 4):
        super().__init__()
        
        self.input_mlp = MLP(D_a, H)
        encoders = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=M,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoders, num_layers=E)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(output_size=T)
        self.output_mlp = MLP(H, D_l)


        
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