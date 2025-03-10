from typing import Tuple
import torch
from torch import nn, Tensor

class LinearProjector(nn.Module):
    def __init__(self, encoder_dim: int, embed_dim: int, approx_target_length: int = 150):
        super().__init__()
        self.downsample_factor = encoder_dim // approx_target_length
        # initial projection from encoder dimensions
        self.linear1 = nn.Linear(in_features=encoder_dim, out_features=embed_dim, bias=False)
        # combined downsampling and final projection
        self.downsample_proj = nn.Linear(self.downsample_factor * embed_dim, embed_dim, bias=False)
        
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
        hidden_states = self.linear1(audio_features)
        # prepare for downsampling
        hidden_states, L = self._pad_sequence(hidden_states)
        # reshape and downsample
        hidden_states = hidden_states.view(hidden_states.shape[0], L // self.downsample_factor, -1)
        hidden_states = self.downsample_proj(hidden_states)
        # final projection
        return hidden_states