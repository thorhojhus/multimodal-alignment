import os
import torch
from transformers import AutoModel
from transformers import WhisperModel
from transformers import ClapModel
from transformers import Wav2Vec2Model, Wav2Vec2BertModel
from transformers import MimiModel
from enum import Enum


class EncoderEnum(Enum):
    WHISPER = "openai/whisper-base"
    WHISPER_LARGE = "openai/whisper-large-v3"
    WHISPER_TURBO = "openai/whisper-large-v3-turbo"
    HVISKE = "syvai/hviske-v2"
    CLAP = "laion/clap-htsat-unfused"
    WAV2VEC_BASE_960h = "facebook/wav2vec2-base-960h"
    WAV2VEC_LARGE_xlsr_53 = "facebook/wav2vec2-large-xlsr-53"
    WAV2VEC_LARGE = "facebook/wav2vec2-large"
    WAV2VECBERT = "facebook/w2v-bert-2.0"
    WAV2VEC2_LV60_ESPEAK_CV_FT = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    MIMI = "kyutai/mimi"
    BICODEC = "BiCodec"

class EncoderWrapper(AutoModel):
    """As the different audio encoders have different configs, and different input types, wrap the encoders"""
    def __init__(self, model: str):
        self.encoder_type = EncoderEnum(model)
        self.audio_encoder = self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize/load the encoder."""

        match self.encoder_type:
            case EncoderEnum.WHISPER | EncoderEnum.WHISPER_TURBO | EncoderEnum.HVISKE | EncoderEnum.WHISPER_LARGE:
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
            case EncoderEnum.WAV2VEC_BASE_960h | EncoderEnum.WAV2VEC_LARGE | EncoderEnum.WAV2VEC_LARGE_xlsr_53 | EncoderEnum.WAV2VEC2_LV60_ESPEAK_CV_FT:
                encoder = Wav2Vec2Model.from_pretrained(
                    self.encoder_type.value,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map=None
                ).to("cuda")
            case EncoderEnum.WAV2VECBERT:
                encoder = torch.compile(Wav2Vec2BertModel.from_pretrained(
                    self.encoder_type.value,
                    torch_dtype=torch.bfloat16,
                    device_map=None
                ).to("cuda"), dynamic=True)
            case EncoderEnum.MIMI:
                encoder = MimiModel.from_pretrained(
                    self.encoder_type.value,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map="cuda"
                )
            case EncoderEnum.BICODEC:
                from models.bicodec_wav2vec_wrapper import BiCodecWrapper
                encoder = BiCodecWrapper(
                    bicodec_dir=f"{os.environ["HF_HOME"]}/BiCodec",
                    wav2vec_model="facebook/wav2vec2-large-xlsr-53").to("cuda", torch.bfloat16)


        # freeze encoder params
        for param in encoder.parameters():
            param.requires_grad = False

        
        return encoder

    def get_proj_dim(self):
        """Return the projection dimension we expect the Audio Projection Layer to receive."""
        match self.encoder_type:
            case EncoderEnum.WHISPER | EncoderEnum.WHISPER_TURBO | EncoderEnum.HVISKE | EncoderEnum.WHISPER_LARGE:
                return self.audio_encoder.config.d_model
            case EncoderEnum.CLAP: 
                return self.audio_encoder.config.hidden_size # hidden_size is output shape _before_ the _models_ projection layer, which we do not use in favor of our own.
            case EncoderEnum.WAV2VEC_BASE_960h | EncoderEnum.WAV2VECBERT |EncoderEnum.WAV2VEC_LARGE | EncoderEnum.WAV2VEC_LARGE_xlsr_53 | EncoderEnum.WAV2VEC2_LV60_ESPEAK_CV_FT:
                return self.audio_encoder.config.hidden_size
            case EncoderEnum.MIMI:
                # For Mimi, no projection dimension needed as we handle the tokens directly with MimiTokenizer
                return 32  # Number of codebooks in Mimi
            case EncoderEnum.BICODEC:
                return 1

    def prepare_for_projection(self, audio_encoded: torch.Tensor) -> torch.Tensor:
        """Prepare the encoder for projection."""
        match self.encoder_type:
            case EncoderEnum.WHISPER | EncoderEnum.WHISPER_TURBO | EncoderEnum.HVISKE | EncoderEnum.WHISPER_LARGE:
                return audio_encoded
            case EncoderEnum.CLAP:
                B, D, _, _ = audio_encoded.size()
                return audio_encoded.permute(0, 2, 3, 1).reshape(B, -1, D)
            case EncoderEnum.WAV2VEC_BASE_960h | EncoderEnum.WAV2VECBERT | EncoderEnum.WAV2VEC_LARGE | EncoderEnum.WAV2VEC_LARGE_xlsr_53 | EncoderEnum.WAV2VEC2_LV60_ESPEAK_CV_FT:
                return audio_encoded
            case EncoderEnum.MIMI:
                # Mimi encoder outputs audio codes in shape [batch_size, 32, seq_len]
                # No transformation needed as MimiTokenizer handles this shape directly
                return audio_encoded
            case EncoderEnum.BICODEC:
                return audio_encoded

    def __call__(self, *args, **kwargs):
        if isinstance(args[0], dict):
            return self.audio_encoder(**args[0])
        return self.audio_encoder(*args, **kwargs)

if __name__ == "__main__":
    encoder = EncoderWrapper(EncoderEnum.BICODEC.value)
    print(encoder)
    import numpy as np
    x = np.random.randn(2, 32000)
    print(encoder(x))
    # encoder = EncoderWrapper(EncoderEnum.WHISPER.value)
    # print(encoder)
    # encoder = EncoderWrapper(EncoderEnum.CLAP.value)
    # print(encoder)
