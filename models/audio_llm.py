import transformers
import torch
from torch import nn
from dataclasses import dataclass

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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        self.audio_token_str = "<|audio|>"
        self.add_special_token()
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token_str)
        self.downsample_factor = config.downsample_factor

    def create_lm(self, config):
        lm = transformers.AutoModelForCausalLM.from_pretrained(
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

    def adjust_masks_and_labels(self, input_ids, attention_mask, labels, audio_token_positions, audio_embeds):
        B, T = input_ids.shape
        D = audio_embeds.shape[1]
        
        new_attention_masks = []
        new_labels_list = []
        
        for batch_idx in range(B):
            positions = [pos[1] for pos in audio_token_positions if pos[0] == batch_idx]
            num_audio_tokens = len(positions)
            new_seq_len = T + num_audio_tokens * (D - 1)
            
            new_attention = torch.zeros(new_seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
            new_labels = torch.full((new_seq_len,), -100, dtype=labels.dtype, device=labels.device)
            
            orig_idx = 0
            new_idx = 0
            audio_pos_iter = iter(sorted(positions))
            next_audio_pos = next(audio_pos_iter, None)
            
            while orig_idx < T:
                if orig_idx == next_audio_pos:
                    audio_duration = D
                    new_attention[new_idx:new_idx + audio_duration] = 1
                    new_labels[new_idx:new_idx + audio_duration] = -100
                    next_audio_pos = next(audio_pos_iter, None)
                    new_idx += audio_duration
                    orig_idx += 1
                else:
                    new_attention[new_idx] = attention_mask[batch_idx, orig_idx]
                    if orig_idx < T - 1:
                        new_labels[new_idx] = labels[batch_idx, orig_idx]
                    new_idx += 1
                    orig_idx += 1
            
            new_attention_masks.append(new_attention)
            new_labels_list.append(new_labels)
        
        new_attention_mask = torch.stack(new_attention_masks)
        new_labels = torch.stack(new_labels_list)
        return new_attention_mask, new_labels

    def forward(self, input_ids: torch.Tensor, audio_features: torch.Tensor, 
                attention_mask: torch.Tensor = None, labels: torch.Tensor = None):
        with torch.no_grad():
            audio_encoded = self.audio_encoder(audio_features).last_hidden_state
        audio_embeds = self.projector(audio_encoded)
        B, D, _ = audio_embeds.shape
        
        input_embeds = self.lm.get_input_embeddings()(input_ids)
        audio_token_positions = (input_ids == self.audio_token_id).nonzero(as_tuple=False).tolist()
        
        # adjust attention_mask and labels before modifying input_embeds
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if labels is None:
            labels = input_ids.clone()

        # handle labels by shifting and setting padding to -100
        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
        
        # adjust masks and labels for audio embeddings
        new_attention_mask, new_labels = self.adjust_masks_and_labels(
            input_ids, attention_mask, labels, audio_token_positions, audio_embeds
        )
        
        combined_embeds = []
        for batch_idx in range(B):
            batch_positions = [pos[1] for pos in audio_token_positions if pos[0] == batch_idx]
            current_embeds = input_embeds[batch_idx]
            for pos in sorted(batch_positions, reverse=True):
                audio_embed = audio_embeds[batch_idx]
                before = current_embeds[:pos]
                after = current_embeds[pos+1:]
                current_embeds = torch.cat([before, audio_embed, after], dim=0)
            combined_embeds.append(current_embeds)
        
        combined_embeds = torch.stack(combined_embeds)

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