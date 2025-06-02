import torchinfo
import torch
from torch import nn, Tensor
from torch.nn import functional as F

class ReLUSquared(nn.Module):
    """Squared ReLU activation from Primer paper."""
    def forward(self, x):
        return F.relu(x) ** 2

class MLP(nn.Module):
    """Multi-layer perceptron with RMS normalization."""
    def __init__(self, input_dim: int, output_dim: int, act, hidden: int = 256):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden),
            nn.RMSNorm(output_dim),
        ])
        
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

class MDHAAttention(nn.Module):
    """Multi-DConv-Head Attention from Primer paper.
    Applies depthwise convolution after Q, K, V projections.
    """
    def __init__(self, d_model, n_heads, kernel_size=3, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V (no bias as per Primer)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Depthwise convolutions after projections
        # groups=d_model means each channel has its own kernel
        self.dconv_q = nn.Conv1d(d_model, d_model, kernel_size, 
                                 padding=kernel_size-1, groups=d_model, bias=False)
        self.dconv_k = nn.Conv1d(d_model, d_model, kernel_size, 
                                 padding=kernel_size-1, groups=d_model, bias=False)
        self.dconv_v = nn.Conv1d(d_model, d_model, kernel_size, 
                                 padding=kernel_size-1, groups=d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, key_padding_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 1. Linear projections
        Q = self.w_q(x)  # [B, L, D]
        K = self.w_k(x)
        V = self.w_v(x)
        
        # 2. Apply depthwise convolutions
        # Transpose for conv1d: [B, L, D] -> [B, D, L]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Apply causal convolution (only look at past tokens)
        Q = self.dconv_q(Q)[:, :, :seq_len]
        K = self.dconv_k(K)[:, :, :seq_len]
        V = self.dconv_v(V)[:, :, :seq_len]
        
        # Transpose back: [B, D, L] -> [B, L, D]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 3. Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 4. Scaled dot-product attention (uses PyTorch's optimized implementation)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True if mask is None else False
        )
        
        # 5. Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.dropout(self.w_o(attn_output))

class PrimerEncoderLayer(nn.Module):
    """Transformer encoder layer with all Primer modifications:
    - MDHA (Multi-DConv-Head Attention)
    - Squared ReLU activation
    - No bias in linear layers
    - Pre-norm architecture
    """
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        
        # Primer uses 4x expansion ratio for FFN
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        
        # MDHA instead of standard multi-head attention
        self.self_attn = MDHAAttention(d_model, nhead, kernel_size=3, dropout=dropout)
        
        # Feed-forward network with no bias
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        
        # Pre-norm layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = ReLUSquared()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-norm multi-head attention
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Pre-norm feed-forward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class DepthwiseDownsampler(nn.Module):
    """
    Depthwise separable convolution for temporal downsampling.
    """
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
        self.act = ReLUSquared()
        
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

class PrimerTransformerProjector(nn.Module):
    """Audio feature projector with Primer-based transformer architecture.
    
    Processes variable-length audio features through an efficient transformer
    and downsamples temporally to a fixed output dimension.
    
    Incorporates all Primer optimizations:
    - Squared ReLU activations
    - MDHA (Multi-DConv-Head Attention) 
    - No bias in linear layers
    - Pre-norm architecture
    """
    def __init__(
            self,
            encoder_dim: int,
            embed_dim: int,
            hidden_dim: int = 1024,
            attention_heads: int = 8,
            encoder_layers: int = 8,
            dropout: float = 0.1,
            downsample_factor: int = 8
        ):
        super().__init__()

        # Squared ReLU activation
        self.act = ReLUSquared()

        # Input projection
        self.input_mlp = MLP(encoder_dim, hidden_dim, act=self.act)
        
        # Stack of Primer encoder layers
        self.transformer_encoder = nn.ModuleList([
            PrimerEncoderLayer(
                d_model=hidden_dim,
                nhead=attention_heads,
                dim_feedforward=4 * hidden_dim,  # 4x expansion
                dropout=dropout
            )
            for _ in range(encoder_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.RMSNorm(hidden_dim)
        
        # Temporal downsampling using depthwise separable convolution
        self.temporal_downsample = DepthwiseDownsampler(
            hidden_dim, 
            downsample_factor=downsample_factor
        )
        
        # Output projection
        self.output_mlp = MLP(hidden_dim, embed_dim, act=self.act)
        
        # Initialize weights following Primer
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, audio_features: Tensor) -> Tensor:
        """
        Args:
            audio_features: [batch, seq_len, encoder_dim]
        Returns:
            [batch, seq_len//8, embed_dim]
        """
        # Input projection: [batch, seq_len, encoder_dim] -> [batch, seq_len, hidden_dim]
        features = self.input_mlp(audio_features)
        
        # Apply transformer encoder layers
        for layer in self.transformer_encoder:
            features = layer(features)
        
        # Final normalization
        features = self.final_norm(features)
        
        # Temporal downsampling: [batch, seq_len, hidden_dim] -> [batch, seq_len//8, hidden_dim]
        features = features.permute(0, 2, 1)  # [batch, hidden_dim, seq_len]
        features = self.temporal_downsample(features)  # [batch, hidden_dim, seq_len//8]
        features = features.permute(0, 2, 1)  # [batch, seq_len//8, hidden_dim]
        
        # Output projection: [batch, seq_len//8, hidden_dim] -> [batch, seq_len//8, embed_dim]
        features = self.output_mlp(features)

        return features

if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    ENCODER_DIM = 2048
    EMBED_DIM = 2304
    
    # Create model
    model = PrimerTransformerProjector(
        encoder_dim=ENCODER_DIM,
        embed_dim=EMBED_DIM,
        hidden_dim=1024,
        attention_heads=8,
        encoder_layers=8,
        dropout=0.1,
        downsample_factor=8
    ).to(device, torch.bfloat16)
    
    # Show model summary
    print("\nModel Summary:")
    torchinfo.summary(model, input_size=(2, 1600, ENCODER_DIM), dtypes=[torch.bfloat16])

    # Test with different sequence lengths
    test_lengths = [100, 101, 102, 103, 107, 109]
    
    print("Testing AudioProjector with 8x downsampling:\n")
    for seq_len in test_lengths:
        audio_features = torch.randn(2, seq_len, ENCODER_DIM).to(device, torch.bfloat16)
        output = model(audio_features)
        
        print(f"Input shape:  {audio_features.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Downsampling: {seq_len} → {seq_len//8}\n")
    