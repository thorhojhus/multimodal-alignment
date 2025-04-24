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

class AudioTokenizer(nn.Module):
    """Whisper-based audio tokenizer as an nn.Module."""
    def __init__(
        self,
        vocab_size: int = 16384,
        embed_dim: int = 512,
        hidden_stride: int = 5,
        d_model: int = 512,
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
        self.hidden_stride = hidden_stride
        self.d_model = d_model
        
        # head layers
        try:
            self.head = nn.Sequential(
                nn.Linear(
                    self.d_model * hidden_stride,
                    vocab_size,
                    bias=False,
                    device=self.device,
                    dtype=dtype
                ),
                nn.LayerNorm(vocab_size, device=self.device, dtype=dtype)                
            )
        except Exception:
            self.head = nn.Sequential(
                nn.Linear(
                    self.d_model * hidden_stride,
                    vocab_size,
                    bias=False,
                    device=self.device,
                    dtype=dtype
                ),
                nn.LayerNorm(vocab_size).to(device=self.device)          
            )  
        
        # audio embedding
        self.audio_embedding = AudioEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            device=self.device,
            dtype=dtype
        )

    def tokenize(self, logits: Tensor) -> Tensor:
        """Convert logits to token probabilities using softmax."""
        return softmax(logits, dim=-1)

    def forward(self, features: Tensor) -> Tensor:
        """
        Forward pass of the audio tokenizer.
        
        Args:
            features (Tensor): Whisper encoder features [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor: Embedded audio tokens [batch_size, num_tokens, hidden_size]
        """
        
        if self.hidden_stride > 1:
            n, w, d = features.shape
            assert w % self.hidden_stride == 0, "Sequence length must be divisible by hidden_stride"
            features = features.reshape(n, w // self.hidden_stride, self.hidden_stride * d)
        
        logits = self.head(features)  # [batch_size, seq_len/hidden_stride, vocab_size]
        
        tokens = self.tokenize(logits)  # [batch_size, seq_len/hidden_stride, vocab_size]
        
        embeds = self.audio_embedding(tokens)  # [batch_size, num_tokens, hidden_size]
        return embeds

if __name__ == "__main__":
    # ensure CUDA is available
    assert torch.cuda.is_available(), "CUDA is required for this script"

    # initialize the tokenizer
    tokenizer = AudioTokenizer(
        vocab_size=16384,
        embed_dim=512,
        hidden_stride=5,
        encoder_name="openai/whisper-tiny",
    )

    # mock audio input: assuming preprocessed mel spectrograms
    input_features = torch.randn(1, 80, 3000, dtype=torch.bfloat16, device="cuda")

    embeds = tokenizer(input_features)
    print(embeds.shape)  # Expected: [1, 600, 512] (3000 / 5 = 600 tokens)
    print(embeds.dtype)  # Expected: torch.bfloat16
    print(embeds.device)  # Expected: cuda:0