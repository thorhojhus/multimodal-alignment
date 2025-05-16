from typing import Tuple
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
    # freqs: [seq_len, head_dim]
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # Need to reshape freqs to be broadcastable with q and k
    freqs = freqs.unsqueeze(0).unsqueeze(1) # (1, 1, seq_len, head_dim)
    
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
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        freqs = self.rotary_emb(seq_len, device=hidden_states.device) # freqs: [seq_len, head_dim]
        q, k = apply_rotary_pos_emb(q, k, freqs)
        
        attention_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
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
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        x = self.ln1(hidden_states)
        x = self.attention(x)
        x = self.dropout(x)
        hidden_states = x + residual
        
        residual = hidden_states
        x = self.ln2(hidden_states)
        x = self.feed_forward(x)
        x = self.dropout(x)
        hidden_states = x + residual
        
        return hidden_states

class SpeechAdapter(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 3072, 
                 ffn_dim: int = 4096, llm_embed_dim: int = 4096,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        self.transformer_layer = RotaryTransformerLayer(
                hidden_size=hidden_dim,
                intermediate_size=ffn_dim,
                num_heads=24,
                dropout_rate=dropout_rate
            )
        
        self.linear = nn.Linear(hidden_dim, llm_embed_dim)
        self.ln_out = nn.LayerNorm(llm_embed_dim)
        
    def forward(self, audio_features: Tensor) -> Tensor:
        hidden_states = audio_features.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        
        hidden_states = self.transformer_layer(hidden_states)
            
        hidden_states = self.linear(hidden_states)
        hidden_states = self.ln_out(hidden_states)
        
        return hidden_states

if __name__ == "__main__":
    torch.manual_seed(42)
    
    adapter = SpeechAdapter(
        input_dim=1536, 
        hidden_dim=3072, 
        ffn_dim=4096, 
        llm_embed_dim=4096,
        dropout_rate=0.1
    )
    
    param_count = sum(p.numel() for p in adapter.parameters() if p.requires_grad) # Count only trainable
    print(f"SpeechAdapter has {param_count:,} trainable parameters (~{param_count/1e6:.2f}M)")
    
    batch_size = 2
    seq_len = 100
    input_dim = 1536
    input_tensor = torch.randn(batch_size, seq_len, input_dim).cuda() # Move to CUDA if available
    adapter = adapter.cuda() # Move model to CUDA
    
    print(f"Input shape: {input_tensor.shape}")
    output = adapter(input_tensor)
    print(f"Output shape: {output.shape}")
    
    print("\nSample of output (first 5 values of first sequence in batch):")
    print(output[0, 0, :5])