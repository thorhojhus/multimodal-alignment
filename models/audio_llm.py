import transformers
from models.whisper_modified import WhisperModel
import torch
from torch import nn, Tensor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
from liger_kernel.transformers import AutoLigerKernelForCausalLM

@dataclass
class AudioLLMConfig:
    downsample_factor: int = 8
    lm_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    whisper_model_name: str = "openai/whisper-base"

class Projector(nn.Module):
    def __init__(self, whisper_dim: int, embed_dim: int, downsample_factor: int):
        super().__init__()
        self.downsample_factor = downsample_factor
        # initial projection from whisper dimensions
        self.linear1 = nn.Linear(in_features=whisper_dim, out_features=embed_dim, bias=False)
        self.relu = nn.ReLU()
        # combined downsampling and final projection
        self.downsample_proj = nn.Linear(downsample_factor * embed_dim, embed_dim, bias=False)
        
    def _pad_sequence(self, hidden_states: Tensor) -> Tuple[Tensor, int]:
        # pad if sequence length is not divisible by downsample_factor
        B, L, D = hidden_states.shape
        if L % self.downsample_factor != 0:
            pad_size = self.downsample_factor - (L % self.downsample_factor)
            padding = hidden_states.new_zeros(B, pad_size, D)
            hidden_states = torch.cat([hidden_states, padding], dim=1)
        return hidden_states, hidden_states.shape[1]
        
    def forward(self, audio_features: Tensor) -> Tensor:
        # initial projection
        hidden_states = self.relu(self.linear1(audio_features))
        # prepare for downsampling
        hidden_states, L = self._pad_sequence(hidden_states)
        # reshape and downsample
        hidden_states = hidden_states.view(hidden_states.shape[0], L // self.downsample_factor, -1)
        # final projection
        return self.downsample_proj(hidden_states)

class AudioLLMModel(nn.Module):
    def __init__(self, config: AudioLLMConfig):
        super().__init__()
        self.config = config
        # create model components
        self.audio_encoder = self._create_audio_encoder()
        self.lm = self._create_lm()
        self.projector = self._create_projector()
        self.tokenizer = self._setup_tokenizer()
        # setup special tokens
        self.audio_token_str = "<|audio|>"
        self._add_special_token()
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token_str)
        self.pad_token_id = self.tokenizer.pad_token_id

    def _create_audio_encoder(self) -> nn.Module:
        # initialize whisper encoder with flash attention (only encoder to GPU)
        whisper = WhisperModel.from_pretrained(
            self.config.whisper_model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=None)
        encoder = whisper.encoder.to("cuda")
        # freeze encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    def _create_lm(self) -> nn.Module:
        # initialize language model with flash attention
        lm = AutoLigerKernelForCausalLM.from_pretrained(
            self.config.lm_model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        # freeze lm parameters
        for param in lm.parameters():
            param.requires_grad = False
        return lm

    def _create_projector(self) -> nn.Module:
        # get dimensions for projector
        whisper_dim = self.audio_encoder.config.d_model
        embed_dim = self.lm.get_input_embeddings().weight.shape[1]
        return Projector(whisper_dim, embed_dim, self.config.downsample_factor).to("cuda", torch.bfloat16)

    def _setup_tokenizer(self) -> transformers.PreTrainedTokenizer:
        # initialize tokenizer from language model
        return transformers.AutoTokenizer.from_pretrained(self.config.lm_model_name)

    def _add_special_token(self) -> None:
        # add audio token if not in vocabulary
        if self.audio_token_str not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_token_str]})

    def _get_audio_positions(self, input_ids: Tensor) -> List[Tensor]:
        # find positions of audio tokens in each sequence
        return [(row == self.audio_token_id).nonzero(as_tuple=True)[0] for row in input_ids]

    def _adjust_insertion_indices(self, audio_positions: List[Tensor]) -> List[Tensor]:
        # adjust indices for audio token insertion
        return [
            pos - torch.arange(len(pos), device=pos.device) if pos.numel() > 0
            else torch.tensor([], dtype=torch.long, device=pos.device)
            for pos in audio_positions
        ]

    def _get_valid_positions(self, input_ids: Tensor) -> List[Tensor]:
        # get positions of non-audio tokens
        return [(row != self.audio_token_id).nonzero(as_tuple=True)[0] for row in input_ids]

    def _process_audio_features(self, audio_features: Tensor) -> Tensor:
        # encode and project audio features
        with torch.no_grad():
            audio_encoded = self.audio_encoder(audio_features).last_hidden_state
        return self.projector(audio_encoded)

    def _combine_embeddings(
        self, text_embeds: Tensor, audio_embeds: Tensor, 
        insertion_indices: List[Tensor], labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # combine text and audio embeddings for each sample
        combined_embeds_list, combined_labels_list, combined_attention_mask_list = [], [], []
        B, D, _ = audio_embeds.shape

        for i in range(B):
            embeds = text_embeds[i]
            lbls = labels[i] if labels is not None else None
            att_mask = attention_mask[i] if attention_mask is not None else None

            if insertion_indices[i].numel() > 0:
                idx = insertion_indices[i][0].item()
                # insert audio embeddings at specified position
                combined_embeds = torch.cat([embeds[:idx], audio_embeds[i], embeds[idx:]], dim=0)
                
                if lbls is not None:
                    # mark audio positions with -100 in labels
                    audio_labels = torch.full((D,), -100, dtype=lbls.dtype, device=lbls.device)
                    combined_labels = torch.cat([lbls[:idx], audio_labels, lbls[idx:]], dim=0)
                    combined_labels_list.append(combined_labels)

                if att_mask is not None:
                    # set attention mask for audio embeddings
                    audio_mask = torch.ones(D, dtype=att_mask.dtype, device=att_mask.device)
                    combined_att_mask = torch.cat([att_mask[:idx], audio_mask, att_mask[idx:]], dim=0)
                    combined_attention_mask_list.append(combined_att_mask)
            else:
                combined_embeds = embeds
                if lbls is not None:
                    combined_labels_list.append(lbls)
                if att_mask is not None:
                    combined_attention_mask_list.append(att_mask)
            
            combined_embeds_list.append(combined_embeds)

        return combined_embeds_list, combined_labels_list, combined_attention_mask_list

    def _pad_sequences(
        self, combined_lists: Tuple[List[Tensor], List[Tensor], List[Tensor]]
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # pad sequences to uniform length
        embeds_list, labels_list, attention_mask_list = combined_lists
        combined_embeds = torch.nn.utils.rnn.pad_sequence(embeds_list, batch_first=True, padding_value=0)
        
        combined_labels = None
        if labels_list:
            combined_labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            
        combined_attention_mask = None
        if attention_mask_list:
            combined_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            
            # adjust attention mask based on trailing -100s in labels
            if combined_labels is not None:
                for i in range(combined_labels.shape[0]):
                    # find the positions of trailing -100s
                    trailing_mask = torch.flip(combined_labels[i] == -100, [0])
                    # find the first False (non -100) position from the end
                    try:
                        first_non_trailing = torch.where(~trailing_mask)[0][0].item()
                        # set attention mask to 0 for trailing -100s, but don't include the position of last non-100
                        if first_non_trailing > 0:  # only mask if there are actual trailing -100s
                            combined_attention_mask[i, -first_non_trailing:] = 0
                    except IndexError:
                        pass
            
        return combined_embeds, combined_labels, combined_attention_mask
    
    def save_projector(self, save_path: str) -> None:
        # save only the projector weights
        torch.save(self.projector.state_dict(), save_path)
        
    def load_projector(self, load_path: str) -> None:
        # load only the projector weights
        self.projector.load_state_dict(torch.load(load_path, weights_only=True))

    def _prepare_inputs(
        self, input_ids: Tensor, attention_mask: Optional[Tensor], labels: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        # prepare input tensors for model
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if labels is None:
            labels = input_ids.clone()

        audio_positions = self._get_audio_positions(input_ids)
        insertion_indices = self._adjust_insertion_indices(audio_positions)
        valid_positions = self._get_valid_positions(input_ids)

        # remove audio tokens and pad sequences
        new_input_ids = [input_ids[i][pos] for i, pos in enumerate(valid_positions)]
        new_labels = [labels[i][pos] for i, pos in enumerate(valid_positions)]
        new_attention_mask = [attention_mask[i][pos] for i, pos in enumerate(valid_positions)]

        new_input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=self.pad_token_id)
        new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=-100)
        new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
        return new_input_ids, new_attention_mask, new_labels, insertion_indices

    def forward(
        self, input_ids: Tensor, audio_features: Tensor,
        attention_mask: Optional[Tensor] = None, labels: Optional[Tensor] = None
    ) -> Any:
        # prepare input tensors
        new_input_ids, new_attention_mask, new_labels, insertion_indices = self._prepare_inputs(
            input_ids, attention_mask, labels
        )
        
        # process audio and text features
        audio_embeds = self._process_audio_features(audio_features)
        text_embeds = self.lm.get_input_embeddings()(new_input_ids)
        
        # combine embeddings and get padded sequences
        combined_lists = self._combine_embeddings(
            text_embeds, audio_embeds, insertion_indices, new_labels, new_attention_mask
        )
        combined_embeds, combined_labels, combined_attention_mask = self._pad_sequences(combined_lists)
        # forward pass through language model
        return self.lm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels
        )

    def generate(
        self, input_ids: Tensor, audio_features: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_new_tokens: int = 100,
        **generation_kwargs: Any
    ) -> Tensor:
        # prepare inputs for generation
        new_input_ids, new_attention_mask, _, insertion_indices = self._prepare_inputs(
            input_ids, attention_mask, None
        )
        
        # process audio and text features
        audio_embeds = self._process_audio_features(audio_features)
        text_embeds = self.lm.get_input_embeddings()(new_input_ids)
        
        # combine embeddings and get padded sequences
        combined_lists = self._combine_embeddings(
            text_embeds, audio_embeds, insertion_indices, attention_mask=new_attention_mask
        )
        combined_embeds, _, combined_attention_mask = self._pad_sequences(combined_lists)

        # prepare generation configuration
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.pad_token_id,
            **generation_kwargs
        }

        # generate text completions
        return self.lm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            **generation_config
        )
    
if __name__ == "__main__":
    B = 2
    T = 1024

    config = AudioLLMConfig()
    model = AudioLLMModel(config)

    # create a dummy batch where some positions are the audio token.
    input_ids = torch.randint(0, 49152, (B, T)).to("cuda", torch.int64)
    input_ids[0, 1] = model.audio_token_id
    input_ids[1, 2] = model.audio_token_id
    
    audio_features = torch.randn(B, 80, 1500).to("cuda", torch.bfloat16)
    attention_mask = torch.ones(B, T, device="cuda", dtype=torch.int64)
    labels = input_ids.clone()

    outs = model.forward(input_ids, audio_features, attention_mask, labels)
    #gens = model.generate(input_ids, audio_features, attention_mask, max_new_tokens=100)

    print(outs)
    #print(gens)
