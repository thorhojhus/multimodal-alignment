from typing import Tuple
import torch
from torch import nn, Tensor
import math

# Rotary position embedding implementation
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Rotary self-attention module
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
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(seq_len).unsqueeze(0).unsqueeze(0), \
                  self.rotary_emb(seq_len).unsqueeze(0).unsqueeze(0)
        cos, sin = cos[:, :, :seq_len], sin[:, :, :seq_len]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_probs = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(context)
        
        return output

# Feed-forward network for Transformer
class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

# Transformer layer with rotary attention
class RotaryTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int = 24):
        super().__init__()
        self.attention = RotaryAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        # Self-attention with layer norm and residual connection
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed-forward with layer norm and residual connection
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

# Speech adapter as described in the document
class SpeechAdapter(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 3072, ffn_dim: int = 4096, llm_embed_dim: int = 4096):
        super().__init__()
        # Convolution layer with kernel size 3 and stride 2 to reduce frame length to 80ms
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # Rotary Transformer layer
        self.transformer = RotaryTransformerLayer(
            hidden_size=hidden_dim,
            intermediate_size=ffn_dim,
            num_heads=24
        )
        
        # Linear layer to map to LLM embedding dimension
        self.linear = nn.Linear(hidden_dim, llm_embed_dim)

        self.ln_out = nn.LayerNorm(llm_embed_dim)

        
    def forward(self, audio_features: Tensor) -> Tensor:
        # audio_features shape: [batch_size, seq_len, input_dim]
        
        # Apply convolution (need to transpose for conv1d)
        hidden_states = audio_features.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # [batch_size, seq_len/2, hidden_dim]
        
        # Apply transformer layer
        hidden_states = self.transformer(hidden_states)
        
        # Apply final linear projection and layer norm
        hidden_states = self.linear(hidden_states)
        hidden_states = self.ln_out(hidden_states)

        
        return hidden_states


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create an instance of the SpeechAdapter
    adapter = SpeechAdapter(input_dim=1536, hidden_dim=3072, ffn_dim=4096, llm_embed_dim=4096)
    
    # Print model size
    param_count = sum(p.numel() for p in adapter.parameters())
    print(f"SpeechAdapter has {param_count:,} parameters (~{param_count/1e6:.2f}M)")
    
    # Generate a random input tensor
    # Batch size: 2, Sequence length: 100, Feature dimension: 1536 (speech encoder output dim)
    batch_size = 2
    seq_len = 100
    input_dim = 1536
    input_tensor = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass through the adapter
    output = adapter(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output sequence length: ~{seq_len/2} (after conv downsampling)")
    
    # Print sample of the output
    print("\nSample of output (first 5 values of first sequence in batch):")
    print(output[0, 0, :5])