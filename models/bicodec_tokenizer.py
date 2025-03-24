from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import init

class AudioEmbedding(torch.nn.Embedding):
    """Embedding layer for audio tokens, similar to VisualEmbedding in Ovis."""
    def forward(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        if audio_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(audio_tokens)
        return torch.matmul(audio_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()

class BiCodecTokenizer(nn.Module):
    """BiCodec audio tokenizer for integration with LLMs. Tokenizes the semantic tokens into LLM embeddings.
    """
    def __init__(
        self,
        vocab_size: int = 8192, # bicodec has 8k semantic tokens
        embed_dim: int = 512, # should be llm embed dim
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        # set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
    
        # configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # audio embedding for each token
        self.audio_embedding = AudioEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            device=self.device,
            dtype=dtype
        )

        # nn.LayerNorm(vocab_size, device=self.device, dtype=dtype)
        

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass of the BiCodec tokenizer.
        
        Args:
            tokens (Tensor): BiCodec semantic codes [batch_size, seq_len (20 ms)]
            
        Returns:
            Tensor: Embedded audio tokens [batch_size, seq_len, embed_dim]
        """
        _, seq_len = tokens.shape

        outs = self.audio_embedding(tokens)
        
        outs = outs.permute(0,2,1) # swap last two dimensions
        outs = nn.AdaptiveAvgPool1d(seq_len//10)(outs)
        outs = outs.permute(0,2,1) # swap back

        return outs

if __name__ == "__main__":
    # ensure CUDA is available
    assert torch.cuda.is_available(), "CUDA is required for this script"

    # initialize the tokenizer
    tokenizer = BiCodecTokenizer(
        vocab_size=2048,
        embed_dim=512,
    )

    # mock mimi audio codes with shape [batch_size, 32, seq_len]
    # where 32 is the number of codebooks and seq_len is the sequence length
    batch_size = 2
    seq_len = 74  # example from the model card
    input_codes = torch.randint(0, 8192, (batch_size, seq_len), dtype=torch.int64).cuda() 

    # process through the tokenizer
    embeds = tokenizer(input_codes)
    
    print(f"Input shape: {input_codes.shape}")  # Expected: [2, 32, 74]
    print(f"Output shape: {embeds.shape}")      # Expected: [2, 74, 512]
    print(f"Output dtype: {embeds.dtype}")      # Expected: torch.bfloat16
    print(f"Output device: {embeds.device}")    # Expected: cuda:0
