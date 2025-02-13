import torch
from transformers import AutoModel
from transformers import WhisperModel
from transformers import ClapModel


class EncoderWrapper(AutoModel):
    """As the different audio encoders have different configs, and different input types, wrap the encoders"""
    def __init__(self, model: str):
        self.model_id = model
        self.audio_encoder = self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize/load the encoder."""

        if "whisper" in self.model_id or "hviske" in self.model_id:
            whisper = WhisperModel.from_pretrained(
                self.encoder_type.value,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map=None
            )
            encoder = whisper.encoder.to("cuda")
        elif "clap" in self.model_id:
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
        if "whisper" in self.model_id or "hviske" in self.model_id:
            return self.audio_encoder.config.d_model
        elif "clap" in self.model_id: 
            return self.audio_encoder.config.hidden_size

    def prepare_for_projection(self, audio_encoded: torch.Tensor) -> torch.Tensor:
        """Prepare the encoder for projection."""
        if "whisper" in self.model_id or "hviske" in self.model_id:
            return audio_encoded
        elif "clap" in self.model_id:
            B, D, _, _ = audio_encoded.size()
            return audio_encoded.permute(0, 2, 3, 1).reshape(B, -1, D)

    def __call__(self, *args, **kwargs):
        return self.audio_encoder(*args, **kwargs)
