import transformers
import torch
from torch import nn
from dataclasses import dataclass
from liger_kernel.transformers import AutoLigerKernelForCausalLM
@dataclass
class AudioLLMConfig:
    whisper_dim: int = 512
    embed_dim: int = 960
    inter_dim: int = 512
    downsample_factor: int = 6

class Projector(nn.Module):
    def __init__(self, whisper_dim, embed_dim, inter_dim, downsample_factor=6):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.ln_pre = nn.RMSNorm(whisper_dim)
        self.w1 = nn.Linear(whisper_dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, embed_dim, bias=False)
        self.ln_post = nn.RMSNorm(embed_dim)
        if downsample_factor:
            self.downsample = nn.Linear(downsample_factor * embed_dim, embed_dim, bias=False)
        else:
            self.downsample = None
        
    def forward(self, x):
        x = self.ln_pre(x)
        x = self.w1(x)
        x = self.w2(x)
        x = self.ln_post(x)
        
        if self.downsample is not None:
            B, T, D = x.shape
            # Ensure T is divisible by downsample_factor (preprocess if needed)
            x = x.reshape(B, T // self.downsample_factor, self.downsample_factor * D)
            x = self.downsample(x)
        return x

class AudioLLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm = self.create_lm(config)
        self.projector = self.create_projector(config)
        self.audio_encoder = self.create_audio_encoder(config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        self.audio_token_str = "<|audio|>"
        self.add_special_token()
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token_str)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.downsample_factor = config.downsample_factor

    def create_lm(self, config):
        # cannot be used with torch compile, should be config in the future
        lm = AutoLigerKernelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-360M-Instruct", 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        for param in lm.parameters():
            param.requires_grad = False
        return lm

    def create_projector(self, config):
        return Projector(config.whisper_dim, config.embed_dim, config.inter_dim, config.downsample_factor).to("cuda", torch.bfloat16)

    def create_audio_encoder(self, config):
        whisper = transformers.WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base.en", 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        encoder = whisper.model.encoder
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    def add_special_token(self):
        if self.audio_token_str not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_token_str]})
            self.lm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)

    def adjust_masks_and_labels(self, input_ids, attention_mask, labels, audio_embeds):
        D = audio_embeds.shape[1]
        audio_mask = input_ids == self.audio_token_id
        
        counts = torch.where(audio_mask, D, 1).flatten()  # (B*T,)
        
        # expand masks and labels using flattened counts
        new_attention_mask = torch.repeat_interleave(attention_mask.flatten(), counts).view(attention_mask.size(0), -1)
        new_labels = torch.repeat_interleave(labels.flatten(), counts).view(labels.size(0), -1)
        
        # create expanded audio mask and apply label masking
        expanded_audio_mask = torch.repeat_interleave(audio_mask.flatten(), counts).bool().view(audio_mask.size(0), -1)
        new_labels[expanded_audio_mask] = -100
        
        return new_attention_mask, new_labels

    def forward(self, input_ids: torch.Tensor, audio_features: torch.Tensor, 
                attention_mask: torch.Tensor = None, labels: torch.Tensor = None):
        with torch.no_grad():
            audio_encoded = self.audio_encoder(audio_features).last_hidden_state
        audio_embeds = self.projector(audio_encoded)
        B, D, _ = audio_embeds.shape
        
        input_embeds = self.lm.get_input_embeddings()(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if labels is None:
            labels = input_ids.clone()

        # handle labels with padding tokens
        labels.masked_fill_(labels == self.pad_token_id, -100)
        
        # adjust masks and labels
        new_attention_mask, new_labels = self.adjust_masks_and_labels(
            input_ids, attention_mask, labels, audio_embeds
        )
        
        # vectorized construction of combined_embeds
        audio_mask = input_ids == self.audio_token_id
        counts = torch.where(audio_mask, D, 1).flatten()
        
        # expand embeddings
        combined_embeds = torch.repeat_interleave(
            input_embeds.flatten(end_dim=1), 
            counts, 
            dim=0
        ).view(input_embeds.size(0), -1, input_embeds.size(-1))
        
        # insert audio embeddings using boolean mask
        audio_positions = torch.repeat_interleave(audio_mask.flatten(), counts).view(audio_mask.size(0), -1)
        combined_embeds[audio_positions] = audio_embeds.flatten(end_dim=1)
        
        outputs = self.lm(inputs_embeds=combined_embeds, attention_mask=new_attention_mask, labels=new_labels)
        return outputs

if __name__ == "__main__":
    B = 2
    T = 1024

    model = AudioLLMModel(AudioLLMConfig())

    input_ids = torch.randint(0, 49152, (B, T)).to("cuda", torch.long)
    input_ids[0, 1] = model.audio_token_id
    input_ids[1, 2] = model.audio_token_id
    
    audio_features = torch.randn(B, 80, 3000).to("cuda", torch.bfloat16)
    attention_mask = torch.ones(B, T, device="cuda", dtype=torch.long)
    labels = input_ids.clone()

    outs = model.forward(input_ids, audio_features, attention_mask, labels)
    print(outs)