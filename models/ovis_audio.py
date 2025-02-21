from typing import Union, Optional

import torch
from torch.nn import init
from torch.nn.functional import softmax, pad
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig, WhisperModel, WhisperProcessor


AUDIO_ATOM_ID = -300
AUDIO_INDICATOR_IDS = [-301, -302, -303, -304, -305]  # Indicators for audio segments


class AudioEmbedding(torch.nn.Embedding):
    """Embedding layer for audio tokens, similar to VisualEmbedding in Ovis."""
    def forward(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        if audio_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(audio_tokens)
        return torch.matmul(audio_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()

class BaseAudioTokenizerConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=16384,
        tokenize_function="softmax",
        sample_rate=16000,
        backbone_config: Optional[Union[PretrainedConfig, dict]] = None,
        hidden_stride: int = 5,  # Downsample by 5 (1500 -> 300)
        hidden_size: int = 512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.sample_rate = sample_rate
        self.backbone_kwargs = {}
        if backbone_config is not None:
            assert isinstance(backbone_config, (PretrainedConfig, dict)), \
                f"expect `backbone_config` to be instance of PretrainedConfig or dict, but got {type(backbone_config)}"
            if not isinstance(backbone_config, PretrainedConfig):
                model_type = backbone_config['model_type']
                backbone_config.pop('model_type')
                backbone_config = AutoConfig.for_model(model_type, **backbone_config)
        self.backbone_config = backbone_config
        self.hidden_stride = hidden_stride
        self.hidden_size = hidden_size  # For embedding compatibility

class BaseAudioTokenizer(PreTrainedModel):
    base_model_prefix = "backbone"
    main_input_name = "input_features"  # Whisper uses input_features (mel spectrograms)
    _processor_class = WhisperProcessor
    _processor_kwargs = {}
    _backbone_class = WhisperModel

    def __init__(self, config: BaseAudioTokenizerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        
        name_or_path = kwargs.get('processor_name_or_path') or kwargs.get('backbone_name_or_path')
        if kwargs.get('train_from_scratch'):
            if not name_or_path:
                raise ValueError("`processor_name_or_path` or `backbone_name_or_path` must be provided when `train_from_scratch=True`")
            self.processor = self._processor_class.from_pretrained(name_or_path, **self._processor_kwargs)
            self.backbone = self._backbone_class.from_pretrained(
                name_or_path,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
                device_map="cuda",
                **self.config.backbone_kwargs
            ).to(device)
            self.config.backbone_config = self.backbone.config
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            if not name_or_path:
                raise ValueError("`processor_name_or_path` must be provided when `train_from_scratch=False`")
            self.processor = WhisperProcessor.from_pretrained(name_or_path)
            self.backbone = AutoModel.from_config(self.config.backbone_config, torch_dtype=dtype).to(device)
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        head_dim = self.config.vocab_size - len(AUDIO_INDICATOR_IDS)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(
                self.backbone.config.d_model * self.config.hidden_stride, head_dim, bias=False
            ).to(dtype=dtype, device=device),
            torch.nn.LayerNorm(head_dim).to(dtype=dtype, device=device)
        )

        self.audio_embedding = AudioEmbedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
            device=device,
            dtype=dtype
        )

    def get_backbone(self):
        return self.backbone

    def get_monitor_tensors(self):
        return {
            "backbone_bottom": self.backbone.encoder.layers[0].self_attn.k_proj.weight,
            "backbone_top": self.backbone.encoder.layers[-1].self_attn.out_proj.weight,
            "head": self.head[0].weight,
            "audio_embedding": self.audio_embedding.weight
        }

    def get_processor(self):
        return self.processor

    def get_audio_embedding(self):
        return self.audio_embedding

    def mock_input(self):
        # Mock 30-second audio at 16kHz
        audio_length = self.config.sample_rate * 30
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, audio_length, dtype=torch.bfloat16, device=device), self.construct_audio_placeholders((1,))

    def get_head(self):
        return self.head

    @staticmethod
    def construct_audio_placeholders(grid):
        # Simplified for 1D audio segmentation
        audio_placeholders = [AUDIO_INDICATOR_IDS[0], AUDIO_ATOM_ID, AUDIO_INDICATOR_IDS[1]]
        if grid[0] > 1:
            for i in range(grid[0]):
                audio_placeholders.append(AUDIO_ATOM_ID)
                if i < grid[0] - 1:
                    audio_placeholders.append(AUDIO_INDICATOR_IDS[2])
        audio_placeholders.append(AUDIO_INDICATOR_IDS[4])
        return audio_placeholders

    def preprocess_audio(self, audio: torch.Tensor, max_segments=9, segment_duration=30.0):
        """
        Preprocess audio by splitting into segments if necessary and converting to mel spectrograms.
        Audio is assumed to be a tensor of shape [audio_length].
        """
        sample_rate = self.config.sample_rate
        audio_length = audio.shape[-1]
        total_duration = audio_length / sample_rate
        device = audio.device
        dtype = torch.bfloat16

        def _preprocess_segment(segment):
            target_length = int(sample_rate * segment_duration)
            if segment.shape[-1] < target_length:
                segment = pad(segment, (0, target_length - segment.shape[-1]), "constant", 0)
            elif segment.shape[-1] > target_length:
                segment = segment[:target_length]
            # Move to CPU and convert to float32 for Whisper processor compatibility
            segment_np = segment.cpu().to(torch.float32).numpy()
            inputs = self.processor(segment_np, sampling_rate=sample_rate, return_tensors="pt")
            # Move back to CUDA and bfloat16
            return inputs["input_features"].to(dtype=dtype, device=device)  # [1, n_mels, 3000] for 30s audio

        def _get_best_segments(duration):
            if duration <= segment_duration:
                return [1]  # Single segment
            num_segments = min(max_segments, int((duration + segment_duration - 1) // segment_duration))
            return [num_segments]

        grid = _get_best_segments(total_duration)
        num_segments = grid[0]
        
        if num_segments == 1:
            input_features = _preprocess_segment(audio)
        else:
            segment_length = audio_length // num_segments
            segments = [audio[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]
            input_features = torch.cat([_preprocess_segment(seg) for seg in segments], dim=0)

        audio_placeholders = self.construct_audio_placeholders(grid)
        return input_features, audio_placeholders

    def tokenize(self, logits):
        # we just use softmax for now
        tokens = softmax(logits, dim=-1)
        return tokens

    def encode(self, input_features):
        with torch.no_grad():
            encoder_outputs = self.backbone.encoder(input_features, return_dict=True)
        features = encoder_outputs.last_hidden_state  # [batch_size, 1500, d_model]

        if self.config.hidden_stride > 1:
            n, l, d = features.shape
            assert l % self.config.hidden_stride == 0, "Sequence length must be divisible by hidden_stride"
            features = features.reshape(n, l // self.config.hidden_stride, self.config.hidden_stride * d)
        
        return features

    def forward(self, input_features) -> torch.Tensor:
        # [BatchSize, n_mels, 3000] -> [BatchSize, #Token, HiddenSize]
        features = self.encode(input_features)  # [batch_size, 300, d_model * hidden_stride]
        logits = self.head(features)  # [batch_size, 300, head_dim]
        tokens = self.tokenize(logits)  # [batch_size, 300, vocab_size - 5]
        
        # Pad with reserved tokens
        batch_size, token_len, _ = tokens.shape
        device = tokens.device
        padding_tensor = torch.zeros(
            size=(batch_size, token_len, len(AUDIO_INDICATOR_IDS)),
            dtype=torch.bfloat16,
            device=device,
            requires_grad=False
        )
        tokens = torch.cat((tokens, padding_tensor), dim=2)

        # apply embedding to get hidden representations
        embeds = self.audio_embedding(tokens)  # [batch_size, 300, hidden_size]
        return embeds

MODEL_TYPE = "whisper_audio_tokenizer"

class WhisperAudioTokenizerConfig(BaseAudioTokenizerConfig):
    model_type = MODEL_TYPE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class WhisperAudioTokenizer(BaseAudioTokenizer):
    config_class = WhisperAudioTokenizerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer"]
    _processor_class = WhisperProcessor
    _processor_kwargs = {}
    _backbone_class = WhisperModel

    def get_monitor_tensors(self):
        return dict(
            backbone_bottom=self.backbone.encoder.layers[0].self_attn.k_proj.weight,
            backbone_top=self.backbone.encoder.layers[-1].self_attn.out_proj.weight,
            head=self.head[0].weight,
            audio_embedding=self.audio_embedding.weight
        )

AutoConfig.register(MODEL_TYPE, WhisperAudioTokenizerConfig)
AutoModel.register(WhisperAudioTokenizerConfig, WhisperAudioTokenizer)

# Usage Example
if __name__ == "__main__":
    # Ensure CUDA is available
    assert torch.cuda.is_available(), "CUDA is required for this script"

    config = WhisperAudioTokenizerConfig(
        backbone_config=AutoConfig.from_pretrained("openai/whisper-tiny"),
        hidden_size=512  # Match Whisper's d_model
    )
    tokenizer = WhisperAudioTokenizer(config, processor_name_or_path="openai/whisper-tiny", train_from_scratch=True)

    # Mock audio: 30 seconds at 16kHz, bfloat16, CUDA
    audio = torch.randn(480000, dtype=torch.bfloat16, device="cuda")
    input_features, placeholders = tokenizer.preprocess_audio(audio)
    embeds = tokenizer(input_features)
    print(embeds.shape)  # Expected: [1, 300, 512]
    print(embeds.dtype)  # Expected: torch.bfloat16
    print(embeds.device)  # Expected: cuda:0