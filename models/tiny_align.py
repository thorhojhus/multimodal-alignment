import torch
import torch.nn as nn

class BridgeFormer(nn.Module):
    def __init__(
        self,
        input_dim,          # ASR feature dimension (Da)
        hidden_dim,         # hidden dimension (H) 
        output_dim,         # LLM embedding dimension (Dl)
        num_heads,          # number of attention heads (M)
        num_layers,         # number of transformer layers (E)
        token_size,         # target sequence length (T)
        dropout=0.1
    ):
        super().__init__()
        
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # transformer encoder layers (no positional encoding as per paper)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # pooling layer combining Conv1D and AdaptiveAvgPool1D
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(token_size)
        self.gelu = nn.GELU()
        
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        
        self.token_size = token_size
        
    def forward(self, x):
        # input shape: [batch_size, seq_len, input_dim]
        
        # input MLP
        x = self.input_mlp(x)  # [batch_size, seq_len, hidden_dim]
        
        # transformer encoder (no positional encoding)
        x = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        
        # pooling with Conv1D and AdaptiveAvgPool1D
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        x = self.conv1d(x)
        x = self.gelu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # [batch_size, token_size, hidden_dim]
        
        # output MLP (2 layers)
        x = self.output_mlp(x)  # [batch_size, token_size, output_dim]
        
        return x

if __name__ == "__main__":
    batch_size = 1
    seq_len = 100
    input_dim = 768    # wav2vec2 feature dimension
    hidden_dim = 256   # as mentioned in paper
    output_dim = 2048  # example LLM embedding dimension
    num_heads = 4      # as mentioned in paper
    num_layers = 4     # as mentioned in paper
    token_size = 30    # default casted token size from paper
    
    model = BridgeFormer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        token_size=token_size
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = model(x)
    print(f"Output shape: {output.shape}")