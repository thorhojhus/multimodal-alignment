import torchinfo
import torch
from torch import nn, Tensor

class MLP(nn.Module):
    """Multi-layer perceptron with RMS normalization and GELU activation."""
    def __init__(self, input_dim: int, output_dim: int, act, hidden: int = 256):
        super().__init__()
        # Group normalization layers
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden),
            nn.RMSNorm(output_dim),
        ])
        
        # Group linear layers (no bias)
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


class DepthwiseDownsampler(nn.Module):
    """Efficient depthwise separable convolution for temporal downsampling."""
    def __init__(self, hidden_dim: int, downsample_factor: int = 8):
        super().__init__()
        
        self.downsample_factor = downsample_factor
        
        # Calculate kernel size and padding for clean convolution
        kernel_size = 2 * downsample_factor - 1
        padding = (kernel_size - 1) // 2
        
        # Depthwise convolution (spatial downsampling)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            stride=downsample_factor,
            padding=padding,
            groups=hidden_dim,
            bias=False
        )
        
        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=1,
            bias=False
        )
        
        # Normalization and activation
        self.norm = nn.RMSNorm(hidden_dim)
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x: [batch, hidden_dim, seq_len]
        Returns:
            [batch, hidden_dim, seq_len // downsample_factor]
        """
        batch_size, channels, seq_len = x.shape
        expected_length = seq_len // self.downsample_factor
        
        # Apply depthwise and pointwise convolutions
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # Ensure exact output length
        if x.size(2) > expected_length:
            x = x[:, :, :expected_length]
        
        # Apply normalization and activation
        x = x.permute(0, 2, 1)  # [B, L, C] for RMSNorm
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.act(x)
        
        return x


class LayeredTransformerProjector(nn.Module):
    """Transformer-based audio feature projector with depthwise downsampling.
    
    Args:
        encoder_dim (int): Input audio feature dimension size
        embed_dim (int): Output feature dimension (LLM embedding dimension)
        hidden_dim (int): Hidden dimension (default: 1024)
        attention_heads (int): Number of attention heads (default: 8)
        encoder_layers (int): Number of encoder layers (default: 8)
        downsample_factor (int): Temporal downsampling factor (default: 8)
    """

    def __init__(
            self,
            encoder_dim: int,
            embed_dim: int,
            hidden_dim: int = 1024,
            attention_heads: int = 8,
            encoder_layers: int = 8,
            downsample_factor: int = 8
        ):
        super().__init__()

        # GELU activation
        self.act = nn.SiLU()

        # Input projection
        self.input_mlp = MLP(encoder_dim, hidden_dim, act=self.act)
        
        # Transformer encoder setup with no bias
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            batch_first=True,
            norm_first=True,  # Pre-norm is generally more stable
            bias=False  # No bias in attention and FFN
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        
        # Pre-downsampling convolution (no bias)
        self.pre_downsample_conv = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        
        # Temporal downsampling using depthwise separable convolution
        self.temporal_downsample = DepthwiseDownsampler(
            hidden_dim, 
            downsample_factor=downsample_factor
        )

        # Output projection
        self.output_mlp = MLP(hidden_dim, embed_dim, act=self.act)
        
        self.downsample_factor = downsample_factor
        self._dimensions = {'D_a': encoder_dim, 'D_l': embed_dim, 'H': hidden_dim}

    def forward(self, audio_features: Tensor) -> Tensor:
        """
        Args:
            audio_features: [batch, seq_len, encoder_dim]
        Returns:
            [batch, seq_len // downsample_factor, embed_dim]
        """
        # Input projection: [batch, seq_len, D_a] -> [batch, seq_len, H]
        features = self.input_mlp(audio_features)
        
        # Transformer encoding: [batch, seq_len, H] -> [batch, seq_len, H]
        features = self.transformer_encoder(features)
        
        # Pre-downsampling conv and activation
        features = features.permute(0, 2, 1)  # [batch, H, seq_len]
        features = self.pre_downsample_conv(features)
        features = self.act(features)
        
        # Temporal downsampling: [batch, H, seq_len] -> [batch, H, seq_len//8]
        features = self.temporal_downsample(features)
        
        # Permute back: [batch, H, seq_len//8] -> [batch, seq_len//8, H]
        features = features.permute(0, 2, 1)
        
        # Output projection: [batch, seq_len//8, H] -> [batch, seq_len//8, D_l]
        features = self.output_mlp(features)

        return features

if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test with different sequence lengths
    print("Testing LayeredTransformerProjector with 8x downsampling:\n")
    
    for seq_len in [800, 1600, 2400]:
        audio_features = torch.randn(2, seq_len, 80).to(device)
        
        model = LayeredTransformerProjector(
            encoder_dim=80, 
            embed_dim=256,
            hidden_dim=1024,
            attention_heads=8,
            encoder_layers=8,
            downsample_factor=8
        ).to(device)
        
        output = model(audio_features)
        
        print(f"Input shape:  {audio_features.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected:     [2, {seq_len//8}, 256]\n")
    
    # Show model summary
    print("Model Summary:")
    model = LayeredTransformerProjector(
        encoder_dim=80, 
        embed_dim=256,
        hidden_dim=512,
        attention_heads=8,
        encoder_layers=6
    ).to(device)
    
    torchinfo.summary(model, input_size=(2, 1600, 80))