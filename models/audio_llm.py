import transformers
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from liger_kernel.transformers import AutoLigerKernelForCausalLM

@dataclass
class AudioLLMConfig:
    whisper_dim: int = 1280
    embed_dim: int = 2048
    inter_dim: int = 4096
    downsample_factor: int = 10

class Projector(nn.Module):
    def __init__(self, whisper_dim: int, embed_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features=whisper_dim, out_features=embed_dim, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states
    

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
        self.audio_avg_pooler = nn.AvgPool1d(config.downsample_factor)
        self.downsample_factor = config.downsample_factor

    def create_lm(self, config):
        lm = AutoLigerKernelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-1.7B-Instruct", 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        for param in lm.parameters():
            param.requires_grad = False
        return lm

    def create_projector(self, config):
        return Projector(config.whisper_dim, config.embed_dim).to("cuda", torch.bfloat16)

    def create_audio_encoder(self, config):
        whisper = transformers.WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo", 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            device_map=None
        )
        encoder = whisper.model.encoder.to("cuda")
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    def add_special_token(self):
        if self.audio_token_str not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_token_str]})

    def adjust_masks_and_labels(self, input_ids, attention_mask, labels):
        """
        Remove the <|audio|> token from input_ids, labels, and attention_mask,
        while recording (adjusted) insertion indices for each sample.
        """
        new_input_ids_list = []
        new_labels_list = []
        new_attention_mask_list = []
        insertion_indices = []

        for i in range(input_ids.size(0)):
            row_ids = input_ids[i]
            row_labels = labels[i]
            row_mask = attention_mask[i]
            # find positions of the audio token in the original row.
            audio_positions = (row_ids == self.audio_token_id).nonzero(as_tuple=True)[0]
            # adjust the indices (new index = original index - number of removed tokens before it)
            if audio_positions.numel() > 0:
                insert_idx = audio_positions - torch.arange(len(audio_positions), device=audio_positions.device)
            else:
                insert_idx = torch.tensor([], dtype=torch.long, device=row_ids.device)
            insertion_indices.append(insert_idx)
            # remove the audio token positions.
            valid_positions = (row_ids != self.audio_token_id).nonzero(as_tuple=True)[0]
            new_input_ids_list.append(row_ids[valid_positions])
            new_labels_list.append(row_labels[valid_positions])
            new_attention_mask_list.append(row_mask[valid_positions])
        
        new_input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        new_labels = torch.nn.utils.rnn.pad_sequence(new_labels_list, batch_first=True, padding_value=-100)
        new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask_list, batch_first=True, padding_value=0)
        return new_input_ids, new_attention_mask, new_labels, insertion_indices

    def forward(self, input_ids: torch.Tensor, audio_features: torch.Tensor, 
                attention_mask: torch.Tensor = None, labels: torch.Tensor = None):
        # process audio features.
        with torch.no_grad():
            audio_encoded = self.audio_encoder(audio_features).last_hidden_state
        audio_embeds = self.projector(audio_encoded)
        # downsample over time (using the entire length of audio_embeds after avg pool)
        audio_embeds = audio_embeds.transpose(1, 2)
        audio_embeds = self.audio_avg_pooler(audio_embeds)
        audio_embeds = audio_embeds.transpose(1, 2)  # (B, D, embed_dim)
        B, D, _ = audio_embeds.shape

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if labels is None:
            labels = input_ids.clone()
        # mask out pad tokens in labels.
        labels.masked_fill_(labels == self.pad_token_id, -100)

        # use the audio token only to get the insertion index, then remove it from the tokens.
        new_input_ids, new_attention_mask, new_labels, insertion_indices = self.adjust_masks_and_labels(
            input_ids, attention_mask, labels
        )

        # get text embeddings from the cleaned input IDs.
        text_embeds = self.lm.get_input_embeddings()(new_input_ids)

        # for each sample, insert the entire audio_embeds block at the recorded index.
        combined_embeds_list = []
        combined_labels_list = []
        combined_attention_mask_list = []
        for i in range(B):
            embeds = text_embeds[i]    # shape (L, hidden)
            lbls = new_labels[i]       # shape (L,)
            att_mask = new_attention_mask[i]  # shape (L,)
            if insertion_indices[i].numel() > 0:
                idx = insertion_indices[i][0].item()
                combined_embeds = torch.cat([embeds[:idx], audio_embeds[i], embeds[idx:]], dim=0)
                # mark the inserted audio positions with -100 in the labels.
                audio_labels = torch.full((D,), -100, dtype=lbls.dtype, device=lbls.device)
                combined_labels = torch.cat([lbls[:idx], audio_labels, lbls[idx:]], dim=0)
                # set the attention mask for the audio embeddings to 1.
                audio_mask_tokens = torch.ones(D, dtype=att_mask.dtype, device=att_mask.device)
                combined_att_mask = torch.cat([att_mask[:idx], audio_mask_tokens, att_mask[idx:]], dim=0)
            else:
                combined_embeds = embeds
                combined_labels = lbls
                combined_att_mask = att_mask
            combined_embeds_list.append(combined_embeds)
            combined_labels_list.append(combined_labels)
            combined_attention_mask_list.append(combined_att_mask)

        # pad the combined sequences to uniform length.
        combined_embeds = torch.nn.utils.rnn.pad_sequence(combined_embeds_list, batch_first=True, padding_value=0)
        combined_labels = torch.nn.utils.rnn.pad_sequence(combined_labels_list, batch_first=True, padding_value=-100)
        combined_attention_mask = torch.nn.utils.rnn.pad_sequence(combined_attention_mask_list, batch_first=True, padding_value=0)

        outputs = self.lm(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask, labels=combined_labels)
        return outputs

if __name__ == "__main__":
    B = 2
    T = 1024

    model = AudioLLMModel(AudioLLMConfig())

    # create a dummy batch where some positions are the audio token.
    input_ids = torch.randint(0, 49152, (B, T)).to("cuda", torch.long)
    input_ids[0, 1] = model.audio_token_id
    input_ids[1, 2] = model.audio_token_id
    
    audio_features = torch.randn(B, 80, 3000).to("cuda", torch.bfloat16)
    attention_mask = torch.ones(B, T, device="cuda", dtype=torch.long)
    labels = input_ids.clone()

    outs = model.forward(input_ids, audio_features, attention_mask, labels)
    print(outs)
