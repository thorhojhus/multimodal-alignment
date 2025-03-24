import torch
import torchaudio
from torch import Tensor
from transformers import AutoProcessor, AutoTokenizer, AutoFeatureExtractor
from transformers import BatchFeature
from datasets import Audio
from dataset.chat_templates import CHATML_TEMPLATE, GEMMA_2, LLAMA3_TEMPLATE, GEMMA_3


class AudioChatDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 lm_model: str="HuggingFaceTB/SmolLM2-360M-Instruct",
                 dataset: str="coral", split="train.clean.100", 
                 tokenizer=None, 
                 processor=None,
                 T=30,
                 config=None):
        self.lm_model_name = lm_model
        self.dataset_name = dataset
        self.config = config
        # Handle multiple datasets separated by commas
        if "," in dataset:
            datasets = []
            dataset_names = dataset.split(",")
            loaded_dataset_names = []
            
            for dataset_name in dataset_names:
                dataset_name = dataset_name.strip()
                loaded_dataset = self._load_single_dataset(dataset_name, split)
                if loaded_dataset is not None:
                    datasets.append(loaded_dataset)
                    loaded_dataset_names.append(dataset_name)
                else:
                    print(f"WARNING: Failed to load dataset: {dataset_name}")
            
            if not datasets:
                raise ValueError(f"No valid datasets found in: {dataset}")
                
            from datasets import concatenate_datasets
            import random
            
            # Keep only essential fields (audio and text) before concatenation
            simplified_datasets = []
            for ds in datasets:
                # Create new dataset with only audio and text columns
                simplified_ds = ds.select_columns(['audio', 'text'])

                simplified_datasets.append(simplified_ds)
                
            # concat and shuffle datasets
            self.dataset = concatenate_datasets(simplified_datasets)
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            self.dataset = self.dataset.select(indices)
            
        else:
            # Single dataset
            self.dataset = self._load_single_dataset(dataset, split)
            if self.dataset is None:
                raise ValueError(f"Failed to load dataset: {dataset}")
            print(f"Loaded dataset {dataset} with {len(self.dataset)} samples")
            
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.tokenizer = tokenizer
        self.T = T # TinyAlign Tokenizer Cutoff/Pad to value.
        self._setup_chat_template()
        # initialize the appropriate processor based on model type
        if processor is not None:
            self.processor = processor
        elif config is not None and hasattr(config, 'model') and hasattr(config.model, 'encoder_model_name') and "mimi" in config.model.encoder_model_name.lower():
            self.processor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        else:
            self.processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
        self.audio_token = "<|audio|>"
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
    
    def _load_single_dataset(self, dataset_name, split):
        """Helper function to load a single dataset"""
        try:
            split_key = split.split(".")[0]  # Handle LibriSpeech style splits
            
            if dataset_name == "coral":
                from dataset.load_coral import coral_dataset_dict
                return coral_dataset_dict[split_key].cast_column("audio", Audio(sampling_rate=16000))
            elif dataset_name == "librispeech":
                from dataset.load_librispeech import librispeech_dataset_dict
                return librispeech_dataset_dict[str(split)].cast_column("audio", Audio(sampling_rate=16000))
            elif dataset_name == "cv_arabic":
                from dataset.common_voice_ar import common_voice_ar_dataset_dict
                return common_voice_ar_dataset_dict[split_key]
            elif dataset_name == "melatech":
                from dataset.load_melatech import melatech_dataset_dict
                return melatech_dataset_dict[split_key].cast_column("audio", Audio(sampling_rate=16000))
            else:
                print(f"Unsupported dataset: {dataset_name}")
                return None
        except KeyError as e:
            print(f"Error loading dataset {dataset_name} with split {split}: Split not found - {e}")
            return None
        except Exception as e:
            print(f"Error loading dataset {dataset_name} with split {split}: {e}")
            return None

    def __len__(self):
        return len(self.dataset)
    
    def _setup_chat_template(self):
        if "SmolLM2" in self.lm_model_name:
            self.tokenizer.chat_template = CHATML_TEMPLATE
        elif "Llama-3" in self.lm_model_name:
            self.tokenizer.chat_template = LLAMA3_TEMPLATE
        elif "gemma-2" in self.lm_model_name:
            self.tokenizer.chat_template = GEMMA_2
        elif "gemma-3" in self.lm_model_name:
            self.tokenizer.chat_template = GEMMA_3
        else:
            print("Custom chat template not found")
            pass

    def _audio_to_features(self, audio_array, audio_sampling_rate):
        
        if "SeamlessM4TFeatureExtractor" in self.processor.__class__.__name__:
            target_sampling_rate = 16000
        elif "Encodec" in self.processor.__class__.__name__:
            target_sampling_rate = self.processor.sampling_rate
        elif "Mimi" in self.processor.__class__.__name__:
            target_sampling_rate = 16000  # Mimi expects 16kHz
        elif "BiCodec" in self.config.model.encoder_model_name:
            target_sampling_rate = 16000
        else:
            try:
                target_sampling_rate = self.processor.feature_extractor.sampling_rate
            except:
                target_sampling_rate = 16000

        # resample if necessary
        if audio_sampling_rate != target_sampling_rate:
            audio_array = torchaudio.functional.resample(
                audio_array, 
                orig_freq=audio_sampling_rate, 
                new_freq=target_sampling_rate
            )

        # Handle Mimi and other models that need normalization
        if "MimiFeatureExtractor" in self.processor.__class__.__name__ or "Mimi" in self.processor.__class__.__name__:
            # Normalize audio to prevent clipping
            audio_array = audio_array / (audio_array.abs().max() + 1e-8)
            
        if ("Wav2Vec2" in self.processor.__class__.__name__):
            # for wav2vec, return raw waveform
            
            if len(audio_array) / audio_sampling_rate > 17: # truncate audio to 17 seconds
                audio_array = audio_array[:audio_sampling_rate*17]
            return audio_array
        elif "MimiFeatureExtractor" in self.processor.__class__.__name__ or "Mimi" in self.processor.__class__.__name__:
            # Process with Mimi feature extractor following the HuggingFace example
            return self.processor(
                raw_audio=audio_array,
                sampling_rate=target_sampling_rate,
                return_tensors="pt"
            )
        elif ("SeamlessM4TFeatureExtractor" in self.processor.__class__.__name__):
            return self.processor(
                audio_array,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
            )
        elif ("Clap" in self.processor.__class__.__name__):
            return self.processor(
                audios=audio_array,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
            ).input_features[0]
        elif ("Whisper" in self.processor.__class__.__name__):
            return self.processor(
                audio=audio_array,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
                truncation=True,
            ).input_features[0]
        elif ("Encodec" in self.processor.__class__.__name__):
            return self.processor(
                raw_audio=audio_array,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
            )["input_values"]
        elif ("BiCodec" in self.config.model.encoder_model_name):
            if len(audio_array) / audio_sampling_rate > 17: # truncate audio to 17 seconds
                audio_array = audio_array[:audio_sampling_rate*17]
            if type(audio_array) is torch.Tensor:
                audio_array = audio_array.numpy()
            audio_array = audio_array.squeeze()
            return  self.processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True, output_hidden_states=True)
        


    def _format_chat(self, example, audio):
        if self.lm_model_name != "gemma":
            return [
                {"role": "system", "content": "You transcribe speech to text. You need to repeat the text once exactly as it is spoken."},
                {"role": "user", "content": f"Transcribe the following text in lower case: {audio}"},
                {"role": "assistant", "content": example}
            ]
        else:
            return [
                {"role": "user", "content": f"Transcribe the following text in lower case: {audio}"},
                {"role": "assistant", "content": example}
            ]
        
    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio = example["audio"]
        text = example["text"].lower()
        # Handle potential missing fields in simplified datasets
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            audio_array = torch.tensor(audio["array"], dtype=torch.float32)
            audio_sampling_rate = audio["sampling_rate"]
        else:
            # For compatibility with datasets that might have a different structure
            print(f"Warning: Unexpected audio format: {type(audio)}")
            if hasattr(audio, "numpy"):
                audio_array = torch.tensor(audio.numpy(), dtype=torch.float32)
                audio_sampling_rate = 16000  # Default to 16kHz if not specified
            else:
                raise ValueError(f"Unsupported audio format: {type(audio)}")


        audio_features = self._audio_to_features(audio_array, audio_sampling_rate)
        # format as chat
        conversation = self._format_chat(text, self.audio_token)
        conversation_alt = self._format_chat(text, text)

        # tokenize conversation with special handling for audio token
        tokenized = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            return_tensors="pt"
        )[0]
        
        tokenized_alt = self.tokenizer.apply_chat_template(
            conversation_alt,
            add_generation_prompt=False,
            return_tensors="pt"
        )[0]

        # Tokenize only the part of the conversation that is part of the assistant.
        assistant_response = self.tokenizer.apply_chat_template(
            [conversation[-1]],
            add_generation_prompt=False,
            return_tensors="pt"
        )[0]
        
        # Label such that only assistance part is considering doing the model pass
        labels = torch.full(tokenized.size(), fill_value=-100)
        labels[-len(assistant_response):] = assistant_response

        alt_labels = torch.full(tokenized_alt.size(), fill_value=-100)
        alt_labels[-len(assistant_response):] = assistant_response
        
        return {
            "input_ids": tokenized,
            "audio_features": audio_features,
            "labels": labels,
            "alt_input_ids": tokenized_alt,
            "alt_labels": alt_labels,
            "text_ids": torch.tensor(self.tokenizer.encode(text)),
        }


def get_audio_positions(input_ids: Tensor, audio_token_id: int) -> list[Tensor]:
    # find positions of audio tokens in each sequence
    return [(row == audio_token_id).nonzero(as_tuple=True)[0] for row in input_ids]

def get_valid_positions(input_ids: Tensor, audio_token_id: int) -> list[Tensor]:
    # get positions of non-audio tokens
    return [(row != audio_token_id).nonzero(as_tuple=True)[0] for row in input_ids]

def adjust_insertion_indices(audio_positions: list[Tensor]) -> list[Tensor]:
    adjusted_indices = []
    
    for positions in audio_positions:
        if positions.numel() > 0:
            # create a sequence of numbers from 0 to len(positions)-1
            shift_values = torch.arange(len(positions), device=positions.device)
            adjusted_indices.append(positions - shift_values)
        else:
            # return an empty tensor if no audio tokens are found
            adjusted_indices.append(torch.tensor([], dtype=torch.long, device=positions.device))

    return adjusted_indices

def prepare_inputs(
    input_ids: Tensor, labels: Tensor | None, tokenizer
) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
    # prepare input tensors for model
    attention_mask = torch.ones_like(input_ids)  # NOTE: This does mask out padding correctly right now.
    if labels is None:
        labels = input_ids.clone()

    audio_token_id = tokenizer.convert_tokens_to_ids("<|audio|>")

    audio_positions = get_audio_positions(input_ids, audio_token_id)
    insertion_indices = adjust_insertion_indices(audio_positions)
    valid_positions = get_valid_positions(input_ids, audio_token_id)

    # remove audio tokens and pad sequences
    new_input_ids = [input_ids[i][pos] for i, pos in enumerate(valid_positions)]
    new_labels = [labels[i][pos] for i, pos in enumerate(valid_positions)]
    new_attention_mask = [attention_mask[i][pos] for i, pos in enumerate(valid_positions)]

    new_input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=-100)
    new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
    
    return new_input_ids, new_attention_mask, new_labels, insertion_indices


def collate_fn(batch, tokenizer, T):
    # handle text inputs
    max_len = max(len(item["input_ids"]) for item in batch)
    max_len_pad = max(len(item["alt_input_ids"]) for item in batch)
    pad_token_id = 0
    input_ids = []
    text_ids = []
    labels = []
    audio_features = []
    alt_input_ids = []
    alt_labels = []
    alt_attention_mask = []
    
    for item in batch:
        # pad input_ids and labels
        pad_len = max_len - len(item["input_ids"])
        alt_pad_len = max_len_pad - len(item["alt_input_ids"])
        
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        audio_features.append(item["audio_features"])
        
        alt_input_ids.append(torch.nn.functional.pad(item["alt_input_ids"], (0, alt_pad_len), value=pad_token_id))
        alt_labels.append(torch.nn.functional.pad(item["alt_labels"], (0, alt_pad_len), value=-100))
        
        # create attention mask for alt_input_ids (1s for tokens, 0s for padding)
        alt_mask = torch.ones(len(item["alt_input_ids"]), dtype=torch.long)
        alt_mask = torch.nn.functional.pad(alt_mask, (0, alt_pad_len), value=0)
        alt_attention_mask.append(alt_mask)

        # pad/cutoff text_ids if using EmbedLoss:
        if len(item["text_ids"]) > T:
            ids = item["text_ids"][:T] # truncate as described in TinyAlign
        else:
            pad_len = T - len(item["text_ids"])      # or pad as described in TinyAlign
            ids = torch.nn.functional.pad(item["text_ids"], (0, pad_len), value=pad_token_id)
        
        text_ids.append(ids)

    text_ids = torch.stack(text_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    alt_input_ids = torch.stack(alt_input_ids)
    alt_labels = torch.stack(alt_labels)
    alt_attention_mask = torch.stack(alt_attention_mask)

    # handle audio features based on model type
    if isinstance(audio_features[0], dict) and "input_values" in audio_features[0]:
        # handling for Mimi and similar models that use feature extractors
        input_values_list = [af["input_values"] for af in audio_features]
        
        # check shape and handle appropriately
        if len(input_values_list[0].shape) == 3:  # [1, 1, samples]
            # pad the sequences in the time dimension
            max_len = max(tensor.size(2) for tensor in input_values_list)
            padded_tensors = []
            
            for tensor in input_values_list:
                pad_size = max_len - tensor.size(2)
                padded = torch.nn.functional.pad(tensor, (0, pad_size, 0, 0, 0, 0), value=0.0)
                padded_tensors.append(padded)
                
            # stack them along batch dimension
            audio_features = torch.cat(padded_tensors, dim=0)
        elif len(input_values_list[0].shape) == 2:  # [1, samples]
            # pad the sequences in the time dimension
            max_len = max(tensor.size(1) for tensor in input_values_list)
            padded_tensors = []
            
            for tensor in input_values_list:
                pad_size = max_len - tensor.size(1)
                padded = torch.nn.functional.pad(tensor, (0, pad_size, 0, 0), value=0.0)
                padded_tensors.append(padded)
                
            # stack them along batch dimension to get [batch, 1, samples]
            stacked = torch.cat(padded_tensors, dim=0)
            audio_features = {"input_values": stacked}
        else:
            # other format, try to handle gracefully
            audio_features = {"input_values": torch.stack(input_values_list)}
    elif isinstance(audio_features[0], torch.Tensor):
        # for wav2vec and other raw waveform models
        if len(audio_features[0].size()) == 3:
            reshaped_features = []
            for feat in audio_features:
                # reshape to T,C
                feat = feat.squeeze(0).transpose(0, 1)
                reshaped_features.append(feat)
            # back to B,C,T    
            audio_features = torch.nn.utils.rnn.pad_sequence(
                    reshaped_features,
                    batch_first=True,
                    padding_value=0.0
            ).transpose(1, 2)
                
        else:
            audio_features = torch.nn.utils.rnn.pad_sequence(
                audio_features, 
                batch_first=True, 
                padding_value=0.0
            )
    elif isinstance(audio_features[0], BatchFeature):
        # for wav2vec2-bert and seamless models
        # extract and pad the features
        features_list = [af.input_features[0] for af in audio_features]  # Remove batch dimension
        audio_features = torch.nn.utils.rnn.pad_sequence(
            features_list,
            batch_first=True,
            padding_value=0.0
        )
        
        # create appropriate attention mask
        max_len = audio_features.size(1)
        attention_mask = torch.zeros((len(features_list), max_len), dtype=torch.long)
        for i, feat in enumerate(features_list):
            attention_mask[i, :feat.size(0)] = 1
            
        audio_features = {
            'input_features': audio_features,
            'attention_mask': attention_mask
        }
    else:
        # for spectrogram-based models (Whisper, CLAP)
        audio_features = torch.stack(audio_features)

    prepared_input_ids, prepared_attention_mask, prepared_labels, prepared_insertion_indices = prepare_inputs(
        input_ids, labels, tokenizer
    )
    
    return {
        "input_ids": prepared_input_ids,
        "audio_features": audio_features,
        "labels": prepared_labels,
        "attention_mask": prepared_attention_mask,
        "insertion_indices": prepared_insertion_indices,
        "alt_input_ids": alt_input_ids,
        "alt_labels": alt_labels,
        "alt_attention_mask": alt_attention_mask,
        "text_ids": text_ids,
    }

if __name__ == "__main__":
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    if "<|audio|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio|>"]})
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    dataset = AudioChatDataset(lm_model="smollm2", tokenizer=tokenizer, processor=processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break