from typing import Tuple, Optional, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: Tensor, k: Tensor, freqs: Tensor) -> Tuple[Tensor, Tensor]:
    freqs = freqs.unsqueeze(0).unsqueeze(1)
    cos = freqs.cos()
    sin = freqs.sin()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 24):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        freqs = self.rotary_emb(seq_len, device=hidden_states.device)
        q, k = apply_rotary_pos_emb(q, k, freqs)
        
        # Apply attention mask if provided
        attn_mask = None
        if attention_mask is not None:
            # Convert boolean/binary mask to attention mask
            attn_mask = attention_mask[:, None, None, :]  # [B, 1, 1, seq_len]
            attn_mask = (1.0 - attn_mask) * -10000.0
        
        attention_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attention_output)
        
        return output

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))

class RotaryTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int = 24, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = RotaryAttention(hidden_size, num_heads)
        self.feed_forward = SwiGLUFFN(hidden_size, intermediate_size, bias=False)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        # Pre-norm attention block
        residual = hidden_states
        x = self.ln1(hidden_states)
        x = self.attention(x, attention_mask=attention_mask)
        x = self.dropout(x)
        hidden_states = x + residual
        
        # Pre-norm FFN block
        residual = hidden_states
        x = self.ln2(hidden_states)
        x = self.feed_forward(x)
        x = self.dropout(x)
        hidden_states = x + residual
        
        return hidden_states

class SpeechAdapter(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 3072, 
                 ffn_dim: int = 4096, llm_embed_dim: int = 4096,
                 dropout_rate: float = 0.1, downsample_factor: int = 4):
        super().__init__()
        self.downsample_factor = downsample_factor
        
        # Use a simple strided convolution for consistent downsampling
        # This ensures output_length = ceil(input_length / downsample_factor)
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0  # No padding for exact downsampling
        )
        self.conv_activation = nn.GELU()
        
        # Pre-normalization
        self.pre_ln = nn.LayerNorm(input_dim)
        
        # Transformer layer
        self.transformer_layer = RotaryTransformerLayer(
            hidden_size=hidden_dim,
            intermediate_size=ffn_dim,
            num_heads=24,
            dropout_rate=dropout_rate
        )
        
        # Output projection
        self.linear = nn.Linear(hidden_dim, llm_embed_dim)
        self.ln_out = nn.LayerNorm(llm_embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize conv layer
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
                
        # Initialize linear layer with smaller std
        nn.init.normal_(self.linear.weight, std=0.02)
        
    def _pad_sequence_for_conv(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor], int]:
        """Pad sequence to be divisible by downsample_factor."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Calculate padding needed
        remainder = seq_len % self.downsample_factor
        if remainder == 0:
            return x, attention_mask, seq_len
            
        pad_length = self.downsample_factor - remainder
        
        # Pad the sequence
        x = F.pad(x, (0, 0, 0, pad_length), value=0.0)
        
        # Pad attention mask if provided
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0.0)
            
        return x, attention_mask, seq_len + pad_length
        
    def _downsample_attention_mask(self, attention_mask: Tensor, from_seq_len: int, to_seq_len: int) -> Tensor:
        """Downsample attention mask using max pooling."""
        if attention_mask is None:
            return None
            
        batch_size = attention_mask.shape[0]
        
        # Ensure we're working with the right sequence length
        attention_mask = attention_mask[:, :from_seq_len]
        
        # Use adaptive max pooling to handle any sequence length
        # This will automatically handle cases where from_seq_len doesn't divide evenly
        attention_mask = attention_mask.unsqueeze(1)  # [B, 1, L]
        attention_mask = F.adaptive_max_pool1d(attention_mask, to_seq_len)
        attention_mask = attention_mask.squeeze(1)  # [B, L']
        
        return attention_mask.float()
        
    def forward(self, audio_features: Tensor, attention_mask: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        """
        Forward pass of the speech adapter.
        
        Returns:
            - If attention_mask is provided: Tuple[Tensor, Optional[Tensor]] - (features, downsampled_mask)
            - If no attention_mask: Tensor - just the features
        """
        batch_size, seq_len, _ = audio_features.shape
        
        # Pre-normalize
        audio_features = self.pre_ln(audio_features)
        
        # Pad sequence if needed for exact downsampling
        audio_features, padded_attention_mask, padded_seq_len = self._pad_sequence_for_conv(audio_features, attention_mask)
        
        # Apply convolution
        hidden_states = audio_features.transpose(1, 2)  # [B, D, L]
        hidden_states = self.conv(hidden_states)
        hidden_states = self.conv_activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # [B, L', D]
        
        # Get new sequence length after convolution
        new_seq_len = hidden_states.shape[1]
        
        # Downsample attention mask if provided
        downsampled_mask = None
        if padded_attention_mask is not None:
            downsampled_mask = self._downsample_attention_mask(padded_attention_mask, padded_seq_len, new_seq_len)
        
        # Apply transformer layer with attention mask
        hidden_states = self.transformer_layer(hidden_states, attention_mask=downsampled_mask)
        
        # Output projection
        hidden_states = self.linear(hidden_states)
        hidden_states = self.ln_out(hidden_states)
        
        # Return based on whether attention_mask was provided
        # This matches the expected interface in audio_llm.py
        if attention_mask is not None:
            return hidden_states, downsampled_mask
        else:
            return hidden_states


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test with different downsample factors and sequence lengths
    test_configs = [
        (2, 100), (4, 100), (8, 100), (10, 100),  # Original tests
        (4, 97), (8, 97), (10, 97),  # Odd sequence lengths
        (4, 123), (8, 200), (10, 333),  # Various lengths
    ]
    
    for downsample_factor, seq_len in test_configs:
        print(f"\nTesting with downsample_factor={downsample_factor}, seq_len={seq_len}")
        
        adapter = SpeechAdapter(
            input_dim=1536, 
            hidden_dim=3072, 
            ffn_dim=4096, 
            llm_embed_dim=4096,
            dropout_rate=0.1,
            downsample_factor=downsample_factor
        ).cuda()
        
        param_count = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"SpeechAdapter has {param_count:,} trainable parameters (~{param_count/1e6:.2f}M)")
        
        batch_size = 2
        input_dim = 1536
        
        # Test WITH attention mask (should return tuple)
        input_tensor = torch.randn(batch_size, seq_len, input_dim).cuda()
        attention_mask = torch.ones(batch_size, seq_len).cuda()
        attention_mask[0, int(0.8 * seq_len):] = 0
        
        result = adapter(input_tensor, attention_mask)
        if isinstance(result, tuple):
            output, out_mask = result
            print(f"With mask - Output shape: {output.shape}, Mask shape: {out_mask.shape}")
        else:
            print(f"ERROR: Expected tuple when attention_mask provided, got {type(result)}")
        
        # Test WITHOUT attention mask (should return just tensor)
        result_no_mask = adapter(input_tensor)
        if isinstance(result_no_mask, Tensor):
            print(f"Without mask - Output shape: {result_no_mask.shape}")
        else:
            print(f"ERROR: Expected tensor when no attention_mask, got {type(result_no_mask)}")