from typing import Tuple
import torch
from torch import nn, Tensor
from liger_kernel.transformers import LigerRMSNorm as RMSNorm
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return LigerSiLUMulFunction.apply(gate, x)
    
class Projector(nn.Module):
    def __init__(self, whisper_dim: int, embed_dim: int, downsample_factor: int):
        super().__init__()
        self.ln = RMSNorm(2 * embed_dim)
        self.downsample_factor = downsample_factor
        # initial projection from whisper dimensions
        self.linear1 = nn.Linear(in_features=whisper_dim, out_features= 2 * embed_dim, bias=False)
        self.act = SwiGLU()
        # combined downsampling and final projection
        self.downsample_proj = nn.Linear(in_features=embed_dim * downsample_factor, 
                                       out_features=embed_dim, 
                                       bias=False)
        
    def _pad_sequence(self, hidden_states: Tensor) -> Tuple[Tensor, int]:
        # pad if sequence length is not divisible by downsample_factor
        B, L, D = hidden_states.shape
        if L % self.downsample_factor != 0:
            pad_size = self.downsample_factor - (L % self.downsample_factor)
            padding = hidden_states.new_zeros(B, pad_size, D)
            hidden_states = torch.cat([hidden_states, padding], dim=1)
        return hidden_states, hidden_states.shape[1]
        
    def forward(self, audio_features: Tensor) -> Tensor:
        #print(audio_features.shape)
        # initial projection and normalization
        hidden_states = self.linear1(audio_features)  # [B, L, 2*embed_dim]
        hidden_states = self.ln(hidden_states)
        hidden_states = self.act(hidden_states)      # [B, L, embed_dim]
        
        # prepare for downsampling
        hidden_states, L = self._pad_sequence(hidden_states)
        B, _, D = hidden_states.shape
        
        # reshape for downsampling: [B, L/downsample, downsample*D]
        hidden_states = hidden_states.view(B, L // self.downsample_factor, self.downsample_factor * D)
        
        # final projection
        hidden_states = self.downsample_proj(hidden_states)
        #print(hidden_states.shape)
        return hidden_states