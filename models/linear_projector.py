from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from liger_kernel.transformers import LigerRMSNorm as RMSNorm

class Projector(nn.Module):
    def __init__(self, whisper_dim: int, embed_dim: int, downsample_factor: int):
        super().__init__()
        self.pre_ln = RMSNorm(whisper_dim)
        self.downsample_factor = downsample_factor
        # initial projection from whisper dimensions
        self.linear1 = nn.Linear(in_features=whisper_dim, out_features=embed_dim, bias=False)
        self.act = nn.SiLU()
        # combined downsampling and final projection
        self.downsample_proj = nn.Linear(downsample_factor * embed_dim, embed_dim, bias=False)
        self.post_ln = RMSNorm(embed_dim)
        
    def _pad_sequence(self, hidden_states: Tensor) -> Tuple[Tensor, int]:
        # pad if sequence length is not divisible by downsample_factor
        B, L, D = hidden_states.shape
        if L % self.downsample_factor != 0:
            pad_size = self.downsample_factor - (L % self.downsample_factor)
            padding = hidden_states.new_zeros(B, pad_size, D)
            hidden_states = torch.cat([hidden_states, padding], dim=1)
        return hidden_states, hidden_states.shape[1]
        
    def forward(self, audio_features: Tensor) -> Tensor:
        # initial projection
        self.pre_ln(audio_features)
        hidden_states = self.act(self.linear1(audio_features))
        # prepare for downsampling
        hidden_states, L = self._pad_sequence(hidden_states)
        # reshape and downsample
        hidden_states = hidden_states.view(hidden_states.shape[0], L // self.downsample_factor, -1)
        hidden_states = self.downsample_proj(hidden_states)
        # final projection
        return self.post_ln(hidden_states)