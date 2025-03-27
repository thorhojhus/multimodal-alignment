from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple, Union, Dict
import typing

import torch
import transformers
import os

from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers.functional import liger_kl_div
from torch import Tensor, nn
from torch.nn import functional as F

from models.encoders import EncoderWrapper
from models.UltravoxProjector import Projector
from models.LinearProjector import LinearProjector
from models.tinyalign import BridgeFormer


class AudioLLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # create model components
        self.lm = self._create_lm()
        self.audio_encoder = EncoderWrapper(config.model.encoder_model_name)
        if self.config.model.audio_tokenizer:
            self.projector = self._create_audio_tokenizer()
        elif self.config.model.encoder_model_name == "kyutai/mimi":
            self.projector = self._create_mimi_tokenizer()
        elif (self.config.model.encoder_model_name == "BiCodec") and (not self.config.model.use_expanded_vocab):
            self.projector = self._create_bicodec_tokenizer()
        else:
            self.projector = self._create_projector(config.model.projector)
        self.tokenizer = self._setup_tokenizer()

        # setup special tokens
        self.audio_token_str = "<|audio|>"
        self._add_special_token()
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token_str)
        
        if self.config.model.use_expanded_vocab:
            # Add audio tokens for discrete BiCodec codes
            self._expand_vocab_with_new_audio_tokens()
            # Update BiCodec tokenizer with audio token base ID if using BiCodec
            if hasattr(self, 'bicodec_expanded_tokenizer'):
                self.bicodec_expanded_tokenizer.audio_token_base_id = self.audio_tokens_id_start
        
        if self.tokenizer.pad_token == self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.alpha = 1.0
        self.beta = 1.0
        self.MSELoss = nn.MSELoss()
        self.CosSim = nn.CosineSimilarity(dim=-1)

    def _expand_vocab_with_new_audio_tokens(self, num_new_audio_tokens: int = 8192):
        # expand vocabulary with new audio tokens
        new_tokens = [f"<|audio_{i}|>" for i in range(num_new_audio_tokens)]
        self.audio_tokens_str = new_tokens
        
        # Add tokens to the tokenizer
        orig_vocab_size = len(self.tokenizer)
        self.tokenizer.add_tokens(new_tokens)
        
        # Store the starting ID for the audio tokens
        self.audio_tokens_id_start = orig_vocab_size
        self.audio_tokens_id_end = orig_vocab_size + num_new_audio_tokens - 1
        
        # Map token strings to their IDs for efficient lookup
        self.audio_tokens_ids = {i: orig_vocab_size + i for i in range(num_new_audio_tokens)}
        
        # Resize the LM embeddings to accommodate the new tokens
        self.lm.resize_token_embeddings(len(self.tokenizer))
        
        # Make the embeddings for LLM trainable
        if not self.config.model.get('freeze_embeddings', False):
            embeds = self.lm.get_input_embeddings()
            embeds.requires_grad = True
            
            # Also unfreeze LM head for generation
            if not self.config.model.get('freeze_lm_head', False):
                lm_head = self.lm.get_output_embeddings()
                if lm_head is not None:
                    lm_head.requires_grad = True


    def _create_bicodec_tokenizer(self):
        # Standard BiCodec tokenizer for embedding approach
        from models.bicodec_tokenizer import BiCodecTokenizer
        embed_dim = self.lm.get_input_embeddings().weight.shape[1]
        return BiCodecTokenizer(vocab_size=8192, 
                                embed_dim=embed_dim,
                                dtype=torch.bfloat16,
                                device="cuda")

    def _create_mimi_tokenizer(self):
        from models.mimi_tokenizer import MimiTokenizer
        embed_dim = self.lm.get_input_embeddings().weight.shape[1]
        return MimiTokenizer(vocab_size=2048, 
                                     embed_dim=embed_dim,
                                     use_hierarchical=self.config.model.get('use_hierarchical', True))

    def _create_audio_tokenizer(self):
        from models.audio_tokenizer import AudioTokenizer
        whisper_dim = self.audio_encoder.get_proj_dim()
        embed_dim = self.lm.get_input_embeddings().weight.shape[1]
        return AudioTokenizer(vocab_size=self.config.model.audio_tokenizer_vocab_size, 
                                     embed_dim=embed_dim, 
                                     hidden_stride=self.config.model.downsample_factor,
                                     d_model=whisper_dim)


    def _create_lm(self) -> nn.Module:
        # set quantization configuration
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if self.config.model.load_in_4bit else None

        if "gemma-3" in self.config.model.lm_model_name:
            from transformers import Gemma3ForCausalLM
            # no liger support for gemma-3 yet, can also not get anything but google/gemma-3-1b-it to work
            lm = Gemma3ForCausalLM.from_pretrained(
                self.config.model.lm_model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                quantization_config=quantization_config
            )
        else:
            # initialize language model with flash attention and liger kernels for memory efficiency
            lm = AutoLigerKernelForCausalLM.from_pretrained(
                self.config.model.lm_model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                quantization_config=quantization_config
            )

        # freeze lm parameters
        for param in lm.parameters():
            param.requires_grad = False
        return lm

    def _create_projector(self, type: Literal["simple", "tinyalign"]) -> Projector:
        # get dimensions for projector
        encoder_dim = self.audio_encoder.get_proj_dim()
        embed_dim = self.lm.get_input_embeddings().weight.shape[1]
        if type == "tinyalign":
            return BridgeFormer(encoder_dim=encoder_dim, token_length=self.config.model.token_length, embed_dim=embed_dim).to("cuda", torch.bfloat16)
        if type == "conv":
            from models.cnn_projector import CNNProjector
            return CNNProjector(encoder_dim, embed_dim, self.config.model.downsample_factor).to("cuda", torch.bfloat16)
        elif type == "linear":
            return LinearProjector(encoder_dim, embed_dim, self.config.model.downsample_factor).to("cuda", torch.bfloat16)
        return Projector(encoder_dim, embed_dim, self.config.model.downsample_factor).to("cuda", torch.bfloat16)

    def _setup_tokenizer(self) -> transformers.PreTrainedTokenizer:
        # initialize tokenizer from language model
        return transformers.AutoTokenizer.from_pretrained(self.config.model.lm_model_name)

    def _add_special_token(self) -> None:
        # add audio token if not in vocabulary
        if self.audio_token_str not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_token_str]})

    def embed_loss(self, audio_embeddings: Tensor, text_embeddings: Tensor):
        # Calculate embedding loss for TinyAlign BridgeFormer (embed_loss)
        embed_loss = self.alpha * self.MSELoss(audio_embeddings, text_embeddings) + self.beta * (1 - self.CosSim(audio_embeddings, text_embeddings)).mean()
        return SimpleNamespace(loss=embed_loss)

    def _process_audio_features(self, audio_features: Union[Tensor, Dict[str, Tensor]]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Process audio features through the encoder and projector.
        
        Args:
            audio_features: Raw audio features
            
        Returns:
            If use_expanded_vocab is False:
                Tensor: Projected audio embeddings
            If use_expanded_vocab is True:
                Tuple[Tensor, Tensor]: (Projected audio embeddings, Discrete token IDs)
        """
        # When using expanded vocab with BiCodec, extract discrete tokens
        if self.config.model.use_expanded_vocab and "BiCodec" in self.config.model.encoder_model_name:
            # Extract discrete tokens using the expanded tokenizer
            discrete_tokens = self.audio_encoder(audio_features)
            
            breakpoint()
            
            return discrete_tokens
        
        # Standard processing pathway (no discrete tokens)
        with torch.no_grad():
            if "mimi" in self.config.model.encoder_model_name:
                audio_encoded = self.audio_encoder(audio_features).audio_codes

            elif "BiCodec" in self.config.model.encoder_model_name:
                audio_encoded = self.audio_encoder(audio_features)
            
            else:
                audio_encoded = self.audio_encoder(audio_features).last_hidden_state

            audio_encoded = self.audio_encoder.prepare_for_projection(audio_encoded)

        return self.projector(audio_encoded)
    
    def _combine_embeddings(
        self, text_embeds: Tensor, audio_embeds: Union[Tensor, Tuple[Tensor, Tensor]], 
        insertion_indices: List[Tensor], labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Combine text and audio embeddings for each sample.
        
        If use_expanded_vocab is True and BiCodec is used, then audio_embeds is a tuple
        containing (projected_embeds, discrete_tokens). In this case, discrete tokens
        will be inserted at the audio token position instead of embeddings.
        """
        combined_embeds_list, combined_labels_list, combined_attention_mask_list = [], [], []
        
        # Check if we're using expanded vocab with discrete tokens
        using_discrete_tokens = (self.config.model.use_expanded_vocab and 
                                "BiCodec" in self.config.model.encoder_model_name and
                                isinstance(audio_embeds, tuple))
        
        if using_discrete_tokens:
            # Unpack the tuple (projected_embeds is kept for backward compatibility)
            projected_embeds, discrete_tokens = audio_embeds
            B = discrete_tokens.shape[0]  # Batch size
        else:
            # Traditional approach with embeddings
            B, D, *_ = audio_embeds.shape 

        for i in range(B):
            embeds = text_embeds[i]
            lbls = labels[i] if labels is not None else None
            att_mask = attention_mask[i] if attention_mask is not None else None

            if insertion_indices[i].numel() > 0:
                idx = insertion_indices[i][0].item()
                
                if using_discrete_tokens:
                    # For discrete tokens: We'll return the original embeddings without audio
                    # The forward method will handle inserting the discrete tokens in the input ids
                    combined_embeds = embeds
                    
                    # Store the insertion index and tokens for this batch item
                    if not hasattr(self, "_audio_token_indices"):
                        self._audio_token_indices = []
                        self._audio_discrete_tokens = []
                    
                    self._audio_token_indices.append((i, idx))
                    self._audio_discrete_tokens.append(discrete_tokens[i])
                    
                    # For labels and attention mask handling, we'll use the expected final sequence length
                    if lbls is not None:
                        # Don't modify labels yet, will be handled in forward
                        combined_labels_list.append(lbls)
                    
                    if att_mask is not None:
                        # Don't modify attention mask yet, will be handled in forward
                        combined_attention_mask_list.append(att_mask)
                        
                else:
                    # Traditional approach: Insert embedded audio at the audio token position
                    D = audio_embeds[i].shape[0]
                    combined_embeds = torch.cat([embeds[:idx], audio_embeds[i], embeds[idx:]], dim=0)
                    
                    if lbls is not None:
                        # Mark audio positions with -100 in labels
                        audio_labels = torch.full((D,), -100, dtype=lbls.dtype, device=lbls.device)
                        combined_labels = torch.cat([lbls[:idx], audio_labels, lbls[idx:]], dim=0)
                        combined_labels_list.append(combined_labels)

                    if att_mask is not None:
                        # Set attention mask for audio embeddings
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

    def forward(
        self, input_ids: Tensor, audio_features: Tensor, insertion_indices: Tensor,
        attention_mask: Optional[Tensor] = None, labels: Optional[Tensor] = None,
        alt_input_ids: Optional[Tensor] = None, alt_attention_mask: Optional[Tensor] = None,
        alt_labels: Optional[Tensor] = None, text_ids: Optional[Tensor] = None, trunced_text_ids: Optional[Tensor] = None,
        audio_wavs: Optional[Tensor] = None, **kwargs
    ) -> Any:
        
        # If embed_loss (TinyAlign) is used:
        if self.config.training.loss=="embed_loss":
            text_embeddings = self.lm.get_input_embeddings()(trunced_text_ids)
            audio_embeddings = self._process_audio_features(audio_features)
            # Handle case where audio_embeddings is a tuple (when using discrete tokens)
            if isinstance(audio_embeddings, tuple):
                audio_embeddings = audio_embeddings[0]  # Use projected embeddings
            return self.embed_loss(audio_embeddings, text_embeddings)

        # Calculate audio reconstruction loss if specified
        if self.config.training.loss == "wav_loss":
            output_wavs = self.generate_tts(self.tokenizer.batch_decode(text_ids, skip_special_tokens=True))
            return self._audio_reconstruction_loss(audio_wavs, output_wavs, "L1")

        # Clear token indices from previous forward pass
        if hasattr(self, "_audio_token_indices"):
            self._audio_token_indices = []
            self._audio_discrete_tokens = []

        # Process audio and text features
        audio_embeds = self._process_audio_features(audio_features)
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        
        # Handling for expanded vocabulary approach
        using_discrete_tokens = (self.config.model.use_expanded_vocab and 
                              "BiCodec" in self.config.model.encoder_model_name)
        
        # Combine embeddings and get padded sequences
        combined_lists = self._combine_embeddings(
            text_embeds, audio_embeds, insertion_indices, labels, attention_mask
        )
        combined_embeds, combined_labels, combined_attention_mask = self._pad_sequences(combined_lists)

        # For discrete tokens with expanded vocab, we need to modify the input_ids
        if using_discrete_tokens:
            # Create new input_ids to replace <|audio|> tokens with discrete tokens
            modified_input_ids = []
            modified_attention_mask = []
            modified_labels = []
            
            # Process each batch item with audio tokens
            for batch_idx, _ in self._audio_token_indices:
                # Get the corresponding discrete tokens for this batch item
                discrete_tokens_idx = [i for i, (idx, _) in enumerate(self._audio_token_indices) if idx == batch_idx]
                
                if not discrete_tokens_idx:
                    # No audio tokens for this batch item
                    modified_input_ids.append(input_ids[batch_idx])
                    if attention_mask is not None:
                        modified_attention_mask.append(attention_mask[batch_idx])
                    if labels is not None:
                        modified_labels.append(labels[batch_idx])
                    continue
                
                # Get the insertion point and discrete tokens
                _, insert_pos = self._audio_token_indices[discrete_tokens_idx[0]]
                audio_tokens = self._audio_discrete_tokens[discrete_tokens_idx[0]]
                
                # Insert discrete token IDs at the audio token position
                new_ids = torch.cat([
                    input_ids[batch_idx][:insert_pos],
                    audio_tokens,
                    input_ids[batch_idx][insert_pos+1:]  # +1 to skip the audio token
                ], dim=0)
                modified_input_ids.append(new_ids)
                
                # Update attention mask if provided
                if attention_mask is not None:
                    new_mask = torch.cat([
                        attention_mask[batch_idx][:insert_pos],
                        torch.ones_like(audio_tokens),
                        attention_mask[batch_idx][insert_pos+1:]
                    ], dim=0)
                    modified_attention_mask.append(new_mask)
                
                # Update labels if provided
                if labels is not None:
                    # Mark audio token positions as -100 in labels (don't predict them)
                    audio_labels = torch.full_like(audio_tokens, -100)
                    new_labels = torch.cat([
                        labels[batch_idx][:insert_pos],
                        audio_labels,
                        labels[batch_idx][insert_pos+1:]
                    ], dim=0)
                    modified_labels.append(new_labels)
            
            # Pad all sequences to same length
            if modified_input_ids:
                padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                    modified_input_ids, batch_first=True, padding_value=self.pad_token_id
                )
                
                padded_attention_mask = None
                if modified_attention_mask:
                    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
                        modified_attention_mask, batch_first=True, padding_value=0
                    )
                
                padded_labels = None
                if modified_labels:
                    padded_labels = torch.nn.utils.rnn.pad_sequence(
                        modified_labels, batch_first=True, padding_value=-100
                    )
                
                # Use input_ids directly instead of embeddings
                if self.config.training.loss == "kl_loss":
                    lm_output = self.lm.forward(
                        input_ids=padded_input_ids,
                        attention_mask=padded_attention_mask,
                    )
                    return self._compute_kl_loss(
                        lm_output=lm_output, 
                        labels=padded_labels, 
                        alt_input_ids=alt_input_ids, 
                        alt_attention_mask=alt_attention_mask,
                        alt_labels=alt_labels
                    )
                
                # Forward pass with input IDs directly
                llm_output = self.lm(
                    input_ids=padded_input_ids,
                    attention_mask=padded_attention_mask,
                    labels=padded_labels
                )
                
                return llm_output

        # Standard approach using embeddings (not discrete tokens)
        # Calculate KL loss if specified
        if self.config.training.loss == "kl_loss":
            lm_output = self.lm.forward(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
            )
            return self._compute_kl_loss(
                lm_output=lm_output, 
                labels=combined_labels, 
                alt_input_ids=alt_input_ids, 
                alt_attention_mask=alt_attention_mask,
                alt_labels=alt_labels
            )
        
        # Forward pass through language model (cross-entropy loss)
        llm_output = self.lm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                labels=combined_labels
            )
        
        return llm_output

    def generate(
        self, input_ids: Tensor, audio_features: Tensor,
        insertion_indices: Tensor, attention_mask: Tensor,
        max_new_tokens: int = 100,
        **generation_kwargs: Any
    ) -> Tensor:
        
        # Clear token indices from previous calls
        if hasattr(self, "_audio_token_indices"):
            self._audio_token_indices = []
            self._audio_discrete_tokens = []
            
        # Process audio and text features
        audio_embeds = self._process_audio_features(audio_features)
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        
        # Check if we're using expanded vocab with discrete tokens
        using_discrete_tokens = (self.config.model.use_expanded_vocab and 
                               "BiCodec" in self.config.model.encoder_model_name and
                               isinstance(audio_embeds, tuple))
                               
        # Combine embeddings and get padded sequences
        combined_lists = self._combine_embeddings(
            text_embeds, audio_embeds, insertion_indices, attention_mask=attention_mask
        )
        combined_embeds, _, combined_attention_mask = self._pad_sequences(combined_lists)
        
        # Prepare generation configuration
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.pad_token_id,
            **generation_kwargs
        }
        
        # For discrete tokens with expanded vocab, we need to modify the input_ids
        if using_discrete_tokens and hasattr(self, "_audio_token_indices") and self._audio_token_indices:
            # Create new input_ids to replace <|audio|> tokens with discrete tokens
            modified_input_ids = []
            modified_attention_mask = []
            
            # Process each batch item with audio tokens
            for batch_idx, _ in self._audio_token_indices:
                # Get the corresponding discrete tokens for this batch item
                discrete_tokens_idx = [i for i, (idx, _) in enumerate(self._audio_token_indices) if idx == batch_idx]
                
                if not discrete_tokens_idx:
                    # No audio tokens for this batch item
                    modified_input_ids.append(input_ids[batch_idx])
                    if attention_mask is not None:
                        modified_attention_mask.append(attention_mask[batch_idx])
                    continue
                
                # Get the insertion point and discrete tokens
                _, insert_pos = self._audio_token_indices[discrete_tokens_idx[0]]
                audio_tokens = self._audio_discrete_tokens[discrete_tokens_idx[0]]
                
                # Insert discrete token IDs at the audio token position
                new_ids = torch.cat([
                    input_ids[batch_idx][:insert_pos],
                    audio_tokens,
                    input_ids[batch_idx][insert_pos+1:]  # +1 to skip the audio token
                ], dim=0)
                modified_input_ids.append(new_ids)
                
                # Update attention mask if provided
                if attention_mask is not None:
                    new_mask = torch.cat([
                        attention_mask[batch_idx][:insert_pos],
                        torch.ones_like(audio_tokens),
                        attention_mask[batch_idx][insert_pos+1:]
                    ], dim=0)
                    modified_attention_mask.append(new_mask)
            
            # Pad all sequences to same length
            if modified_input_ids:
                padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                    modified_input_ids, batch_first=True, padding_value=self.pad_token_id
                )
                
                padded_attention_mask = None
                if modified_attention_mask:
                    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
                        modified_attention_mask, batch_first=True, padding_value=0
                    )
                
                # Generate with input IDs directly
                return self.lm.generate(
                    input_ids=padded_input_ids,
                    attention_mask=padded_attention_mask,
                    **generation_config
                )
                
        # Standard approach: generate with embeddings
        return self.lm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            **generation_config
        )

    def _audio_reconstruction_loss(self, in_audio_wav: Tensor, out_audio_wav: Tensor, loss_type: typing.Literal["L1", "L2"]) -> Tensor:
        """Reconstruction loss for audio."""

        if loss_type == "L1":
            loss = F.l1_loss(in_audio_wav, out_audio_wav)
        elif loss_type == "L2":
            loss = F.mse_loss(in_audio_wav, out_audio_wav)
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented.")
        return SimpleNamespace(loss=loss)

    def _compute_kl_loss(
        self,
        lm_output: transformers.modeling_outputs.CausalLMOutputWithPast,
        labels: Optional[torch.Tensor] = None,
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
            encoder_model_name="openai/whisper-base",
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
