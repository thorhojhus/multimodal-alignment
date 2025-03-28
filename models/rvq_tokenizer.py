import torch
from torch import nn
from torch.nn import functional as F

class VectorQuantization(nn.Module):
    def __init__(self, encoding_dim: int, codebook_size: int, embedding_dim: int, dtype: torch.dtype, commit_loss: int = 0.25, projector: nn.Module | None = None):
        super(VectorQuantization, self).__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commit_loss = commit_loss

        self.require_projection = encoding_dim != embedding_dim
        
        self.maybe_project = projector if self.require_projection else nn.Identity()

        self.codebook = nn.Embedding(codebook_size, embedding_dim, dtype=dtype)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, audio_e: torch.Tensor):
        """Forward Pass through VQ layer"""

        # Project the audio_encoded to the embedding dimension
        audio_e = self.maybe_project(audio_e) # only happens if encoding dim != embedding dim

        batch_size, seq_len, *_ = audio_e.size()
        
        # Calculate the distance between the audio_encoded and the codebook
        d = torch.sum(audio_e**2, dim=2, keepdim=True) \
            + torch.sum(self.codebook.weight**2, dim=1) \
            - 2 * torch.matmul(audio_e, self.codebook.weight.t())
        
        # Find closest encodings
        closest_encoding_indices = torch.argmin(d, dim=2)  # [Batch, Seq]
        closest_encoding = torch.zeros(batch_size, seq_len, self.codebook_size, device=audio_e.device, dtype=audio_e.dtype)
        closest_encoding.scatter_(2, closest_encoding_indices.unsqueeze(2), 1)  # [Batch, Seq, codebook_size]

        # Quantize
        audio_q = torch.matmul(closest_encoding, self.codebook.weight)  # [Batch, Seq, embedding_dim]

        # compute loss for embedding
        commitment_loss = torch.mean((audio_q.detach()-audio_e)**2) + self.commit_loss * torch.mean((audio_q - audio_e.detach()) ** 2)

        # preserve gradients
        audio_q = audio_e + (audio_q - audio_e).detach()

        return audio_q #, closest_encoding_indices, commitment_loss

    def get_quantized_vectors(self, indices):
        quantized = 0
        for k, index in enumerate(indices):
            quantized += F.embedding(index, self.codebooks[k].weight)
        return quantized

if __name__ == "__main__":

    VQ = VectorQuantization(1024, 150)
    audio_encoded = torch.rand(150, 3, 1500, 150)
    audio_quantized, indices, loss = VQ(audio_encoded)
    breakpoint()
    