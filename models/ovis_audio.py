from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.functional import softmax
from transformers import WhisperModel

class AudioEmbedding(nn.Module):
    """Embedding layer for audio tokens."""
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.reset_parameters()

    def forward(self, audio_tokens: Tensor) -> Tensor:
        if audio_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return self.embedding(audio_tokens)
        return torch.matmul(audio_tokens, self.embedding.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.embedding.weight, mean=mean, std=std)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

class WhisperAudioTokenizer(nn.Module):
    """Whisper-based audio tokenizer as an nn.Module."""
    def __init__(
        self,
        vocab_size: int = 16384,
        embed_dim: int = 512,
        hidden_stride: int = 5,
        encoder_name: str = "openai/whisper-tiny",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        # set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        self.encoder = WhisperModel.from_pretrained(
            encoder_name,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map=None,
        ).encoder.to(self.device, torch.bfloat16)

        for param in self.encoder.parameters():
            param.requires_grad = False

        # configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_stride = hidden_stride
        self.d_model = self.encoder.config.d_model
        
        # head layers
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
        
        # audio embedding
        self.audio_embedding = AudioEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            device=self.device,
            dtype=dtype
        )

    def encode(self, input_features: Tensor) -> Tensor:
        """Encode audio features using the Whisper backbone."""
        with torch.no_grad():
            encoder_outputs = self.encoder(input_features, return_dict=True)
        features = encoder_outputs.last_hidden_state  # [batch_size, seq_len, d_model]

        if self.hidden_stride > 1:
            n, l, d = features.shape
            assert l % self.hidden_stride == 0, "Sequence length must be divisible by hidden_stride"
            features = features.reshape(n, l // self.hidden_stride, self.hidden_stride * d)
        
        return features

    def tokenize(self, logits: Tensor) -> Tensor:
        """Convert logits to token probabilities using softmax."""
        return softmax(logits, dim=-1)

    def forward(self, input_features: Tensor) -> Tensor:
        """
        Forward pass of the audio tokenizer.
        
        Args:
            input_features (Tensor): Input mel spectrograms [batch_size, n_mels, time_steps]
            
        Returns:
            Tensor: Embedded audio tokens [batch_size, num_tokens, hidden_size]
        """
        features = self.encode(input_features)  # [batch_size, seq_len/hidden_stride, d_model * hidden_stride]
        
        logits = self.head(features)  # [batch_size, seq_len/hidden_stride, vocab_size]
        
        tokens = self.tokenize(logits)  # [batch_size, seq_len/hidden_stride, vocab_size]
        
        embeds = self.audio_embedding(tokens)  # [batch_size, num_tokens, hidden_size]
        return embeds

if __name__ == "__main__":
    # ensure CUDA is available
    assert torch.cuda.is_available(), "CUDA is required for this script"

    # initialize the tokenizer
    tokenizer = WhisperAudioTokenizer(
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