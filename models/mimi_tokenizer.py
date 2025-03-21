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
    """Mimi audio tokenizer for integration with LLMs.
    
    Handles the unique 32-codebook architecture of Mimi by embedding tokens 
    and processing the temporal dimension efficiently.
    """
    def __init__(
        self,
        vocab_size: int = 2048,
        embed_dim: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        use_hierarchical: bool = True
    ):
        super().__init__()
        
        # set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
    
        # configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.use_hierarchical = use_hierarchical

        # audio embedding for each token
        self.audio_embedding = AudioEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim // 2 if use_hierarchical else embed_dim // 32,
            device=self.device,
            dtype=dtype
        )
        
        # projection layers for hierarchical processing
        if use_hierarchical:
            # Two-stage hierarchical projection
            self.codebook_projection = nn.Linear(
                32 * (embed_dim // 2), 
                embed_dim,
                device=self.device,
                dtype=dtype
            )
        else:
            # Direct projection from all codebooks
            self.codebook_projection = nn.Linear(
                embed_dim, 
                embed_dim,
                device=self.device,
                dtype=dtype
            )

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass of the Mimi tokenizer.
        
        Args:
            tokens (Tensor): Mimi audio codes [batch_size, 32, seq_len]
            
        Returns:
            Tensor: Embedded audio tokens [batch_size, seq_len, embed_dim]
        """
        b, codebooks, seq_len = tokens.shape
        
        if self.use_hierarchical:
            # Reshape to handle all codebooks together for each time step
            # [b, 32, seq_len] -> [b*seq_len, 32]
            reshaped_tokens = tokens.permute(0, 2, 1).reshape(-1, codebooks)
            
            # Embed each token to half embed_dim
            # [b*seq_len, 32] -> [b*seq_len, 32, embed_dim//2]
            token_embeddings = self.audio_embedding(reshaped_tokens)
            
            # Flatten the codebook dimension into the embedding dimension
            # [b*seq_len, 32, embed_dim//2] -> [b*seq_len, 32*embed_dim//2]
            flattened_embeddings = token_embeddings.reshape(b*seq_len, -1)
            
            # Project to final embedding dimension
            # [b*seq_len, 32*embed_dim//2] -> [b*seq_len, embed_dim]
            projected_embeddings = self.codebook_projection(flattened_embeddings)
            
            # Reshape back to [b, seq_len, embed_dim]
            embeds = projected_embeddings.reshape(b, seq_len, self.embed_dim)
            
        else:
            # Alternative approach: Embed each token, then combine and project
            # [b, 32, seq_len] -> [b, seq_len, 32]
            tokens_t = tokens.permute(0, 2, 1)
            
            # Embed each token
            # [b, seq_len, 32] -> [b, seq_len, 32, embed_dim//32]
            token_embeddings = torch.stack([self.audio_embedding(tokens_t[:, :, i]) 
                                           for i in range(codebooks)], dim=2)
            
            # Reshape to combine all codebook embeddings
            # [b, seq_len, 32, embed_dim//32] -> [b, seq_len, embed_dim]
            combined_embeddings = token_embeddings.reshape(b, seq_len, self.embed_dim)
            
            # Project to final embedding space
            embeds = self.codebook_projection(combined_embeddings)
            
        return embeds

if __name__ == "__main__":
    # ensure CUDA is available
    assert torch.cuda.is_available(), "CUDA is required for this script"

    # initialize the tokenizer
    tokenizer = MimiTokenizer(
        vocab_size=2048,
        embed_dim=512,
    )

    # mock mimi audio codes with shape [batch_size, 32, seq_len]
    # where 32 is the number of codebooks and seq_len is the sequence length
    batch_size = 2
    seq_len = 74  # example from the model card
    input_codes = torch.randint(0, 2048, (batch_size, 32, seq_len), device="cuda", dtype=torch.long)

    # process through the tokenizer
    embeds = tokenizer(input_codes)
    
    print(f"Input shape: {input_codes.shape}")  # Expected: [2, 32, 74]
    print(f"Output shape: {embeds.shape}")      # Expected: [2, 74, 512]
    print(f"Output dtype: {embeds.dtype}")      # Expected: torch.bfloat16
    print(f"Output device: {embeds.device}")    # Expected: cuda:0
    
    # Time both approaches
    import time
    
    # Try with non-hierarchical approach
    tokenizer_nh = MimiTokenizer(
        vocab_size=2048,
        embed_dim=512,
        use_hierarchical=False
    )
    
    # Warm-up
    for _ in range(5):
        _ = tokenizer_nh(input_codes)
    
    # Timing
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = tokenizer_nh(input_codes)
    torch.cuda.synchronize()
    end_time = time.time()
    
    nh_time = (end_time - start_time) / 100
    embeds_nh = tokenizer_nh(input_codes)
    print(f"\nNon-hierarchical output shape: {embeds_nh.shape}")  # Expected: [2, 74, 512]
    print(f"Non-hierarchical average time: {nh_time*1000:.3f} ms")

    # Try with hierarchical approach
    tokenizer_h = MimiTokenizer(
        vocab_size=2048,
        embed_dim=512,
        use_hierarchical=True
    )
    
    # Warm-up
    for _ in range(5):
        _ = tokenizer_h(input_codes)
    
    # Timing
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = tokenizer_h(input_codes)
    torch.cuda.synchronize()
    end_time = time.time()
    
    h_time = (end_time - start_time) / 100
    embeds_h = tokenizer_h(input_codes)
    print(f"\nHierarchical output shape: {embeds_h.shape}")  # Expected: [2, 74, 512]
    print(f"Hierarchical average time: {h_time*1000:.3f} ms")
    
    # Compare performance
    speedup = nh_time / h_time if h_time < nh_time else -h_time / nh_time
    faster = "hierarchical" if h_time < nh_time else "non-hierarchical"
    
    print(f"\nThe {faster} approach is {abs(speedup):.2f}x {'faster' if speedup > 0 else 'slower'}")