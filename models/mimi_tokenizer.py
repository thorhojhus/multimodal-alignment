from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.functional import softmax

class AudioEmbedding(torch.nn.Embedding):
    """Embedding layer for audio tokens, similar to VisualEmbedding in Ovis."""
    def forward(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        if audio_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(audio_tokens)
        return torch.matmul(audio_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()

class MimiTokenizer(nn.Module):
    """Whisper-based audio tokenizer as an nn.Module."""
    def __init__(
        self,
        vocab_size: int = 2048,
        embed_dim: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        # set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
    
        # configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # audio embedding
        self.audio_embedding = AudioEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            device=self.device,
            dtype=dtype
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass of the audio tokenizer.
        
        Args:
            features (Tensor): Whisper encoder features [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor: Embedded audio tokens [batch_size, num_tokens, hidden_size]
        """
        
        embeds = self.audio_embedding(tokens)
        return embeds

if __name__ == "__main__":
    # ensure CUDA is available
    assert torch.cuda.is_available(), "CUDA is required for this script"

    # initialize the tokenizer
    tokenizer = MimiTokenizer(
        vocab_size=2048,
        embed_dim=512,
    )

    # mock audio input: assuming preprocessed mel spectrograms
    input_features = torch.randn(1, 80, 3000, dtype=torch.bfloat16, device="cuda")

    embeds = tokenizer(input_features)
    print(embeds.shape)  # Expected: [1, 600, 512] (3000 / 5 = 600 tokens)
    print(embeds.dtype)  # Expected: torch.bfloat16
    print(embeds.device)  # Expected: cuda:0