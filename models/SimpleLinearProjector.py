from typing import Tuple, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class SimpleLinearProjector(nn.Module):
    """
    A simple linear projector for audio features that uses a two-layer MLP
    with average pooling for downsampling.
    """
    def __init__(self, encoder_dim: int, embed_dim: int, downsample_factor: int):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.hidden_dim = 2048
        
        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim * downsample_factor, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, embed_dim),
        )
    
    def forward(self, audio_features: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with attention mask handling.
        
        Args:
            audio_features: Tensor of shape [batch_size, seq_len, encoder_dim]
            attention_mask: Optional mask of shape [batch_size, seq_len]
            
        Returns:
            Tuple of:
                - Projected features: Tensor of shape [batch_size, seq_len/downsample_factor, embed_dim]
                - Downsampled attention mask: Optional tensor of shape [batch_size, seq_len/downsample_factor]
        """
        audio_mask = None
        
        if attention_mask is not None:
            # First apply mask to zero out padding in features
            masked_features = audio_features * attention_mask.unsqueeze(-1)
            
            # Calculate downsampled mask - a position is valid if any token in the window is valid
            B, L = attention_mask.shape
            target_len = L // self.downsample_factor
            audio_mask = F.adaptive_max_pool1d(
                attention_mask.float().unsqueeze(1), 
                target_len
            ).squeeze(1)
            
            # Downsample with average pooling (using the masked features)
            features = F.adaptive_avg_pool1d(
                masked_features.permute(0, 2, 1), 
                masked_features.size(2) // self.downsample_factor
            ).permute(0, 2, 1)
        else:
            # No mask, just regular pooling
            features = F.adaptive_avg_pool1d(
                audio_features.permute(0, 2, 1), 
                audio_features.size(2) // self.downsample_factor
            ).permute(0, 2, 1)
        
        # Reshape for the linear layer
        batch_size, seq_len, encoder_dim = features.shape
        features = features.reshape(batch_size, seq_len, encoder_dim * self.downsample_factor)
        
        # Apply MLP
        projected = self.mlp(features)
        
        return projected, audio_mask