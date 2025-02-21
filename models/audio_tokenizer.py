from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig, WhisperFeatureExtractor, WhisperModel
import torch
from torch import Tensor
from torch.nn import init, functional as F
import numpy as np
from typing import List

# Reserved token IDs for audio
AUDIO_START_TOKEN = 0  # Indicator for start of audio
AUDIO_END_TOKEN = 1    # Indicator for end of audio

class AudioEmbedding(torch.nn.Embedding):
    def forward(self, audio_tokens: Tensor) -> Tensor:
        if audio_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(audio_tokens)
        return torch.matmul(audio_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()

class AudioTokenizerConfig(PretrainedConfig):
    model_type = "audio_tokenizer"

    def __init__(
        self,
        vocab_size=16384,
        backbone_name_or_path="openai/whisper-tiny",
        hidden_stride=5,
        out_dim=384,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.backbone_name_or_path = backbone_name_or_path
        self.hidden_stride = hidden_stride
        self.out_dim = out_dim

class LearnableStrideLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, stride: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.stride_weights = torch.nn.Parameter(torch.ones(stride))
        self.output_transform = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = features.shape
        padded_len = ((seq_len + self.stride - 1) // self.stride) * self.stride
        if padded_len > seq_len:
            padding = torch.zeros(
                batch_size, padded_len - seq_len, hidden_dim,
                device=features.device, dtype=features.dtype
            )
            features = torch.cat([features, padding], dim=1)
        features = features.reshape(batch_size, -1, self.stride, hidden_dim)
        stride_weights = F.softmax(self.stride_weights, dim=0)
        weights = stride_weights.view(1, 1, -1, 1)
        features = (features * weights).sum(dim=2)
        features = self.output_transform(features)
        return features

class AudioTokenizer(PreTrainedModel):
    config_class = AudioTokenizerConfig
    base_model_prefix = "backbone"
    main_input_name = "input_features"
    _feature_extractor_class = WhisperFeatureExtractor
    _backbone_class = WhisperModel

    def __init__(self, config: AudioTokenizerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.feature_extractor = self._feature_extractor_class.from_pretrained(config.backbone_name_or_path)
        self.backbone = self._backbone_class.from_pretrained(
            config.backbone_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        ).encoder.to("cuda", torch.bfloat16)
        
        head_dim = config.vocab_size - 2  # Reserve 2 IDs for start/end tokens
        self.head = torch.nn.Linear(self.backbone.config.d_model, head_dim, bias=False).to("cuda", torch.bfloat16)
        if config.hidden_stride > 1:
            self.stride_layer = LearnableStrideLayer(
                hidden_dim=self.backbone.config.d_model,
                stride=config.hidden_stride
            ).to("cuda", torch.bfloat16)
        self.embeds = AudioEmbedding(config.vocab_size, config.out_dim).to("cuda", torch.bfloat16)
        self.embeds.reset_parameters()
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_backbone(self):
        return self.backbone

    def get_feature_extractor(self):
        return self.feature_extractor

    def preprocess_audio(self, audios: List[np.ndarray], sampling_rate: int = 16000):
        audio_durations = [len(audio) / sampling_rate for audio in audios]
        input_features = self.feature_extractor(
            audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )["input_features"].to("cuda", torch.bfloat16)
        return input_features, audio_durations

    def encode(self, input_features, audio_durations=None):
        with torch.no_grad():
            encoder_outputs = self.backbone(input_features)
        features = encoder_outputs.last_hidden_state
        if audio_durations is not None:
            frames_per_second = 50
            for i in range(features.shape[0]):
                actual_frames = int(audio_durations[i] * frames_per_second)
                features[i, actual_frames:, :] = 0
        if self.config.hidden_stride > 1:
            features = self.stride_layer(features)
        return features

    def tokenize(self, features):
        logits = self.head(features)  # [batch_size, seq_len, vocab_size - 2]
        probs = F.softmax(logits, dim=-1)  # Softmax only
        token_ids = torch.argmax(probs, dim=-1) + 2  # Offset by 2 for reserved tokens
        batch_size, seq_len = token_ids.shape
        start_tokens = torch.full((batch_size, 1), AUDIO_START_TOKEN, dtype=torch.long, device=token_ids.device)
        end_tokens = torch.full((batch_size, 1), AUDIO_END_TOKEN, dtype=torch.long, device=token_ids.device)
        tokens = torch.cat([start_tokens, token_ids, end_tokens], dim=1)  # [batch_size, seq_len + 2]
        return tokens

    def forward(self, input_features, audio_duration=None) -> torch.Tensor:
        features = self.encode(input_features, audio_duration)
        tokens = self.tokenize(features)
        embeds = self.embeds(tokens)
        return embeds

# Register with Transformers
AutoConfig.register("audio_tokenizer", AudioTokenizerConfig)
AutoModel.register(AudioTokenizerConfig, AudioTokenizer)

if __name__ == "__main__":
    config = AudioTokenizerConfig(vocab_size=16384, backbone_name_or_path="openai/whisper-tiny")
    model = AudioTokenizer(config, out_dim=2048)
    audio = [np.random.randn(32000)]  # 2 seconds at 16kHz
    input_features, audio_duration = model.preprocess_audio(audio, sampling_rate=16000)
    embeds = model(input_features, audio_duration=audio_duration)
    
    audios = [np.random.randn(32000), np.random.randn(48000)]  # 2 and 3 seconds
    input_features, audio_durations = model.preprocess_audio(audios, sampling_rate=16000)
    embeds = model(input_features, audio_duration=audio_durations)
    print(embeds.shape)