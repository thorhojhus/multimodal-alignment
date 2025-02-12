import torch
from transformers import AutoModel
from transformers import WhisperModel
from transformers import ClapModel
from typing import Literal
from enum import Enum


class EncoderEnum(Enum):
    WHISPER = "openai/whisper-base"
    CLAP = "laion/clap-htsat-unfused"


class EncoderWrapper(AutoModel):
    """As the different audio encoders have different configs, and different input types, wrap the encoders"""
    def __init__(self, model: str):
        self.encoder_type = EncoderEnum(model)
        self.audio_encoder = self._initialize_encoder()
    
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
                encoder = whisper.encoder.to("cuda")
            case EncoderEnum.CLAP:
                model = ClapModel.from_pretrained(
                    self.encoder_type.value,
                    attn_implementation="eager",
                    torch_dtype=torch.bfloat16,
                    device_map=None
                )
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
                return self.audio_encoder.config.hidden_size # hidden_size is output shape _before_ the _models_ projection layer, which we do not use in favor of our own.

    def __call__(self, *args, **kwargs):
        return self.audio_encoder(*args, **kwargs)

if __name__ == "__main__":
    encoder = EncoderWrapper(EncoderEnum.WHISPER.value)
    print(encoder)
    encoder = EncoderWrapper(EncoderEnum.CLAP.value)
    print(encoder)
