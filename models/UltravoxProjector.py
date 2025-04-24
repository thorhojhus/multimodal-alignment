from typing import Tuple
import torch
from torch import nn, Tensor
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
        
    def _pad_sequence(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor, int]:
        # pad if sequence length is not divisible by downsample_factor
        B, L, D = hidden_states.shape
        if L % self.downsample_factor != 0:
            pad_size = self.downsample_factor - (L % self.downsample_factor)
            padding = hidden_states.new_zeros(B, pad_size, D)
            hidden_states = torch.cat([hidden_states, padding], dim=1)
            
            # Also pad attention mask if provided
            if attention_mask is not None:
                mask_padding = attention_mask.new_zeros(B, pad_size)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=1)
                
        return hidden_states, attention_mask, hidden_states.shape[1]
        
    def forward(self, audio_features: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        # initial projection (don't zero out yet - keep context information for projection)
        audio_features = self.pre_ln(audio_features)
        hidden_states = self.act(self.linear1(audio_features))
        
        # prepare for downsampling with attention mask
        hidden_states, attention_mask, L = self._pad_sequence(hidden_states, attention_mask)
        
        # reshape and downsample
        hidden_states = hidden_states.view(hidden_states.shape[0], L // self.downsample_factor, -1)
        
        # downsample attention mask if provided
        downsampled_mask = None
        if attention_mask is not None:
            # downsample attention mask - a position is valid if any token in the window is valid
            attention_mask = attention_mask.view(attention_mask.shape[0], -1, self.downsample_factor)
            downsampled_mask = (attention_mask.sum(dim=-1) > 0).float()
        
        hidden_states = self.downsample_proj(hidden_states)
            
        # final projection
        projected_features = self.post_ln(hidden_states)
        
        return projected_features, downsampled_mask