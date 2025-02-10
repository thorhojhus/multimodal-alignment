import transformers
from models.whisper_modified import WhisperModel
from models.linear_projector import Projector
import torch
from torch import nn, Tensor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers.functional import liger_kl_div
from torch.nn import functional as F
from types import SimpleNamespace

class AudioLLMModel(nn.Module):
    def __init__(self, config):
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
        if self.tokenizer.pad_token == self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.pad_token_id = self.tokenizer.pad_token_id

    def _create_audio_encoder(self) -> WhisperModel:
        # initialize whisper encoder with flash attention (only encoder to GPU)
        whisper = transformers.WhisperModel.from_pretrained(
            self.config.model.whisper_model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=None)
        encoder = whisper.encoder.to("cuda")
        # freeze encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    def _create_lm(self) -> nn.Module:
        name = self.config.model.lm_model_name.split("/")[-1].split("-")[0].lower()
        
        # set quantization configuration
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # initialize language model with flash attention
        lm = AutoLigerKernelForCausalLM.from_pretrained(
            self.config.model.lm_model_name,
            attn_implementation="flash_attention_2" if name != "gemma" else "eager",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            quantization_config=quantization_config if self.config.model.load_in_4bit else None
        )

        # freeze lm parameters
        for param in lm.parameters():
            param.requires_grad = False
        return lm

    def _create_projector(self) -> Projector:
        # get dimensions for projector
        whisper_dim = self.audio_encoder.config.d_model
        embed_dim = self.lm.get_input_embeddings().weight.shape[1]
        return Projector(whisper_dim, embed_dim, self.config.model.downsample_factor).to("cuda", torch.bfloat16)

    def _setup_tokenizer(self) -> transformers.PreTrainedTokenizer:
        # initialize tokenizer from language model
        return transformers.AutoTokenizer.from_pretrained(self.config.model.lm_model_name)

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
        attention_mask: Optional[Tensor] = None, labels: Optional[Tensor] = None,
        alt_input_ids: Optional[Tensor] = None, alt_attention_mask: Optional[Tensor] = None,
        alt_labels: Optional[Tensor] = None, **kwargs
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

        # calculate KL loss if specified
        if self.config.training.kl_loss:
            lm_output = self.lm.forward(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
            )
            return self._compute_kl_loss(lm_output=lm_output, labels=combined_labels, alt_input_ids=alt_input_ids, alt_labels=alt_labels)
        
        # forward pass through language model (cross-entropy loss)
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

    def _compute_kl_loss(
        self,
        lm_output: transformers.modeling_outputs.CausalLMOutputWithPast,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        alt_input_ids: Optional[torch.Tensor] = None,
        alt_attention_mask: Optional[torch.Tensor] = None,
        alt_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        with torch.no_grad():
            alt_inputs_embeds = self.lm.get_input_embeddings().forward(alt_input_ids)
            alt_lm_output = self.lm.forward(
                inputs_embeds=alt_inputs_embeds,
                attention_mask=alt_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        kl_loss = liger_kl_div(
            F.log_softmax(
                lm_output.logits[labels != -100] / self.config.training.kl_temp, dim=-1,
            ),
            F.softmax(
                alt_lm_output.logits[alt_labels != -100] / self.config.training.kl_temp, dim=-1,
            ),
            reduction="batchmean",
        )
        return SimpleNamespace(loss=kl_loss)
    
if __name__ == "__main__":
    B = 2
    T = 1024
    config = SimpleNamespace(
        model=SimpleNamespace(
            whisper_model_name="openai/whisper-base",
            lm_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
            downsample_factor=10
        ),
        training=SimpleNamespace(
            kl_loss=True
        )
    )
    model = AudioLLMModel(config)

    # create a dummy batch where some positions are the audio token.
    input_ids = torch.randint(0, 49152, (B, T)).to("cuda", torch.int64)
    input_ids[0, 1] = model.audio_token_id
    input_ids[1, 2] = model.audio_token_id
    
    audio_features = torch.randn(B, 80, 3000).to("cuda", torch.bfloat16)
    attention_mask = torch.ones(B, T, device="cuda", dtype=torch.int64)
    labels = input_ids.clone()

    outs = model.forward(input_ids, audio_features, attention_mask, labels)
    #gens = model.generate(input_ids, audio_features, attention_mask, max_new_tokens=100)

    print(outs)
    #print(gens)
