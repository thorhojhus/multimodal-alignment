from torch import nn

class CNNProjector(nn.Module):
    def __init__(
        self,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
        kernel_size: int = 5,
        downsample_factor: int = 4
    ):
        super().__init__()
        
        self.left_padding1 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        
        stride = downsample_factor//2
        self.conv1d_1 = nn.Conv1d(enc_out_dim, 2 * enc_out_dim, kernel_size, stride=stride, padding=0)
        self.conv1d_2 = nn.Conv1d(2 * enc_out_dim, 4 * enc_out_dim, kernel_size, stride=stride, padding=0)
        
        self.bn1 = nn.BatchNorm1d(2 * enc_out_dim, eps=1e-3, momentum=0.99)
        self.bn2 = nn.BatchNorm1d(4 * enc_out_dim, eps=1e-3, momentum=0.99)
        
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        
        self.project = nn.Linear(4 * enc_out_dim, llm_embed_dim)
    
    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.left_padding1(x)
        x = self.conv1d_1(x)
        x = self.bn1(x)
        x = self.silu1(x)

        x = self.left_padding2(x)
        x = self.conv1d_2(x)
        x = self.bn2(x)
        x = self.silu2(x)

        x = x.transpose(1, 2)
        x = self.project(x)

        return x