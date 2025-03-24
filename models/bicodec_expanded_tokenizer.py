from typing import Optional
import torch
from torch import nn, Tensor
import os
import yaml
import sys
from safetensors.torch import load_file

class BiCodecExpandedTokenizer(nn.Module):
    """BiCodec tokenizer that maps discrete audio codes to LLM token IDs.
    When use_expanded_vocab is True, audio is represented as a sequence of token IDs
    rather than embedded at a single <|audio|> token position.
    """
    def __init__(
        self,
        vocab_size: int = 8192,  # bicodec has 8k semantic tokens
        embed_dim: int = 512,    # should be llm embed dim
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        audio_token_base_id: int = None,  # ID of first audio token in the LLM vocab
    ):
        super().__init__()
        
        # set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
    
        # configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.audio_token_base_id = audio_token_base_id

        # Initialize BiCodec encoder for discrete token extraction
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from sparktts.models.bicodec import BiCodec
        
        # Get the pretrained path
        pretrained_bicodec_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/weights/BiCodec/"
        
        # Load the pretrained BiCodec config
        with open(os.path.join(pretrained_bicodec_path, "config.yaml"), "r") as yamlfile:
            bicodec_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
        # Create the BiCodec model
        self.bicodec = BiCodec(bicodec_config)
        
        # Load the pretrained weights
        model_dict = load_file(os.path.join(pretrained_bicodec_path, "model.safetensors"))
        self.bicodec.load_state_dict(model_dict, strict=True)
        
        # Set to eval mode and freeze
        self.bicodec.eval()
        for param in self.bicodec.parameters():
            param.requires_grad = False

    def forward(self, audio: Tensor) -> Tensor:
        """
        Extract token IDs from audio.
        
        Args:
            audio (Tensor): Audio waveform [batch_size, channels, samples]
            
        Returns:
            Tensor: Token IDs to use in the language model [batch_size, seq_len]
        """
        return self.extract_discrete_tokens(audio)
        
    def extract_discrete_tokens(self, audio: Tensor) -> Tensor:
        """
        Extract discrete tokens from audio using BiCodec.
        
        Args:
            audio (Tensor): Audio waveform [batch_size, channels, samples]
            
        Returns:
            Tensor: Discrete token indices [batch_size, seq_len]
        """
        # BiCodec expects audio in the range [-1, 1]
        if torch.max(torch.abs(audio)) > 1.0:
            audio = audio / torch.max(torch.abs(audio))
            
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)  # Add a channel dimension (B, C, T)
            
        # Make sure it's correctly formatted
        assert audio.ndim == 3, f"Audio shape should be (B, C, T), got {audio.shape}"
        
        if audio.size(1) != 1:
            audio = audio[:, 0:1, :]  # Take the first channel
            
        audio = audio.to(self.device)
        
        # Extract discrete token indices
        with torch.no_grad():
            features = self.bicodec.extract_features(audio)
            
            # Get discrete code indices from the quantizer
            _, _, indices = self.bicodec.quantizer(features)  # indices shape: (B, T_q)
            
            # Map to token IDs in the expanded vocab space if audio_token_base_id is provided
            if self.audio_token_base_id is not None:
                token_ids = indices + self.audio_token_base_id
            else:
                token_ids = indices
            
        return token_ids