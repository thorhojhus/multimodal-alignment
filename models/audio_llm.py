from cProfile import label
from math import comb
from sympy.physics.units import au
import transformers
import torch
from torch import nn
from torch import functional as F
from dataclasses import dataclass

from zmq import device

@dataclass
class AudioLLMConfig:
    whisper_dim: int = 512
    embed_dim: int = 960
    inter_dim: int = 512
    downsample_factor: int = 6

class Projector(nn.Sequential):
    def __init__(self, whisper_dim, embed_dim, inter_dim, downsample_factor=6):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.ln_pre = nn.RMSNorm(whisper_dim)
        self.w1 = nn.Linear(whisper_dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, embed_dim, bias=False)
        self.ln_post = nn.RMSNorm(embed_dim)
        self.downsample = nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor) if downsample_factor else None
    
    def forward(self, x):
        x = self.ln_pre(x)
        x = self.w1(x)
        x = self.w2(x)
        x = self.ln_post(x)
        if self.downsample is not None:
            x = x.transpose(1, 2)
            x = self.downsample(x)
            x = x.transpose(1, 2)
        return x
    

class AudioLLMModel():
    def __init__(self, config):
        
        self.lm = self.create_lm(config)
        self.projector = self.create_projector(config)
        self.audio_encoder = self.create_audio_encoder(config)
        self.model = self.create_model(config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        self.audio_token_str = "<|audio|>"
        self.add_special_token()
        
    def create_lm(self, config):
        return transformers.AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct", attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="cuda")

    def create_projector(self, config):
        return Projector(config.whisper_dim, config.embed_dim, config.inter_dim, config.downsample_factor).to("cuda", torch.bfloat16)

    def create_audio_encoder(self, config):
        whisper = transformers.WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en", attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="cuda")
        encoder = whisper.model.encoder
        return encoder

    def create_model(self, config):
        pass

    def add_special_token(self):
        if self.audio_token_str not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_token_str]})
            self.lm.resize_token_embeddings(len(self.tokenizer), mean_resizing=True, pad_to_multiple_of=8)

    def forward(self,
                input_ids: torch.Tensor,
                audio_features: torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None):
        
        audio_encoded = self.audio_encoder(audio_features).last_hidden_state
        audio_embeds = self.projector(audio_encoded)
        input_embeds = self.lm.get_input_embeddings()(input_ids)
        audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token_str)
        audio_token_positions = (input_ids == audio_token_id).nonzero(as_tuple=False)
        
        combined_embeds = []
        for batch_idx, token_idx in audio_token_positions:
            audio_embed = audio_embeds[batch_idx, :, :]
            before_audio = input_embeds[batch_idx, :token_idx + 1, :]
            after_audio = input_embeds[batch_idx, token_idx + 1:, :]
            combined_embeds.append(torch.cat([before_audio, audio_embed, after_audio], dim=0))
        
        combined_embeds = torch.stack(combined_embeds, dim=0)
        
        outputs = self.lm(inputs_embeds=combined_embeds, attention_mask=attention_mask, labels=labels)
        
        return outputs





if __name__ == "__main__":
    B = 2
    T = 1024

    model = AudioLLMModel(AudioLLMConfig())

    input_ids = torch.randint(0, 49152, (B, T)).to("cuda", torch.long)
    # insert audio token
    input_ids[0, 1] = model.tokenizer.convert_tokens_to_ids(model.audio_token_str)
    input_ids[1, 2] = model.tokenizer.convert_tokens_to_ids(model.audio_token_str)
    
    audio_features = torch.randn(B, 80, 3000).to("cuda", torch.bfloat16)

    model.forward(input_ids, audio_features)
