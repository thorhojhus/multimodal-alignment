import torch
from transformers import AutoModel
from transformers import WhisperModel
from transformers import ClapModel
from typing import Literal
from enum import Enum


class EncoderEnum(Enum):
    WHISPER = "openai/whisper-base"
    CLAP = "laion/clap-htsat-fused"


class EncoderWrapper(AutoModel):
    """As the different audio encoders have different configs, and different input types, wrap the encoders"""
    def __init__(self, model: str):
        self.encoder_type = EncoderEnum(model)
        self.audio_encoder = self._initialize_encoder()
        self.sampling_rate = None # gets set in above method call
    
    def _initialize_encoder(self):
        """Initialize/load the encoder."""

        match self.encoder_type:
            case EncoderEnum.WHISPER:
                whisper = WhisperModel.from_pretrained(
                    self.encoder_type.value,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map=None
                )
                self.sampling_rate = whisper.feature_extractor.sampling_rate
                encoder = whisper.encoder.to("cuda")
            case EncoderEnum.CLAP:
                model = ClapModel.from_pretrained(
                    self.encoder_type.value,
                    attn_implementation="eager",
                    torch_dtype=torch.bfloat16,
                    device_map=None
                )
                self.sampling_rate = 16000 # default
                encoder = model.audio_model.audio_encoder.to("cuda")

        # freeze encoder params
        for param in encoder.parameters():
            param.requires_grad = False
        
        return encoder

    def get_proj_dim(self):
        """Return the projection dimension we expect the Audio Projection Layer to receive."""
        match self.encoder_type:
            case EncoderEnum.WHISPER:
                return self.audio_encoder.config.d_model
            case EncoderEnum.CLAP: 
                return self.audio_encoder.config.projection_dim

    def __call__(self, *args, **kwargs):
        return self.audio_encoder(*args, **kwargs)

    
