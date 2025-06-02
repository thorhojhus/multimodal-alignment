import random
import re
from copy import deepcopy

import librosa
import numpy as np
import torch
import torchaudio
from datasets import Audio, Dataset, concatenate_datasets
from phonemizer import phonemize
from scipy.signal import butter, sosfiltfilt
from torch import Tensor
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
)

from config import AppConfig
from dataset.chat_templates import CHATML_TEMPLATE, GEMMA_2, GEMMA_3, LLAMA3_TEMPLATE

class AudioChatDataset(torch.utils.data.Dataset):
    def __init__(self, 
                lm_model: str="HuggingFaceTB/SmolLM2-360M-Instruct",
                dataset: str="coral", 
                split="train.clean.100", 
                tokenizer=None, 
                processor=None,
                T=30,
                config: AppConfig=None):
        """Initialize the AudioChatDataset.
        
        Args:
            lm_model: Name of the language model
            dataset: Dataset name or comma-separated list of dataset names
            split: Dataset split (e.g., train, test)
            tokenizer: Tokenizer instance or None to initialize from lm_model
            processor: Processor instance or None to initialize based on config
            T: TinyAlign Tokenizer Cutoff/Pad value
            config: Configuration object
        """
        # Basic initialization
        self.lm_model_name = lm_model
        self.dataset_name = dataset
        self.config = config
        self.T = T
        self.split = split  # Store split for later use
        
        # Set audio length filtering parameters
        self._init_audio_length_params()
        
        # Load datasets
        if "," in dataset:
            self._load_and_process_multiple_datasets(dataset, split)
        else:
            self._load_and_process_single_dataset(dataset, split)
        
        # Initialize tokenizer and processor
        self._init_tokenizer_and_processor(lm_model, tokenizer, processor)

    def _init_audio_length_params(self):
        """Initialize audio length filtering parameters."""
        # Set default values
        self.min_audio_length = 1.0  # default min length in seconds
        self.max_audio_length = 20.0  # default max length in seconds
        
        # Override with config values if available
        if self.config is not None and hasattr(self.config, 'dataset'):
            if hasattr(self.config.dataset, 'min_audio_length'):
                self.min_audio_length = self.config.dataset.min_audio_length
            if hasattr(self.config.dataset, 'max_audio_length'):
                self.max_audio_length = self.config.dataset.max_audio_length

    def _load_dataset_by_name(self, dataset_name, split):
        """Load a single dataset by name.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split to load
            
        Returns:
            Dataset object or None if loading failed
        """
        try:
            if dataset_name == "coral":
                from dataset.load_coral import coral_dataset_dict
                return coral_dataset_dict[split].cast_column("audio", Audio(sampling_rate=16000))
            elif dataset_name == "librispeech":
                from dataset.load_librispeech import librispeech_dataset_dict
                return librispeech_dataset_dict[split].cast_column("audio", Audio(sampling_rate=16000))
            elif dataset_name == "cv_arabic":
                from dataset.common_voice_ar import common_voice_ar_dataset_dict
                return common_voice_ar_dataset_dict[split].cast_column("audio", Audio(sampling_rate=16000))
            elif dataset_name == "melatech":
                from dataset.load_melatech import melatech_dataset_dict
                return melatech_dataset_dict[split].cast_column("audio", Audio(sampling_rate=16000))
            else:
                print(f"Unsupported dataset: {dataset_name}")
                return None
        except KeyError as e:
            print(f"Error loading dataset {dataset_name} with split {split}: Split not found - {e}")
            return None
        except Exception as e:
            print(f"Error loading dataset {dataset_name} with split {split}: {e}")
            return None

    def _load_and_process_multiple_datasets(self, dataset, split):
        """Load and process multiple datasets.
        
        Args:
            dataset: Comma-separated list of dataset names
            split: Dataset split to load
        """
        datasets = []
        dataset_names = dataset.split(",")
        loaded_dataset_names = []
        
        # Load each dataset
        for dataset_name in dataset_names:
            dataset_name = dataset_name.strip()
            loaded_dataset = self._load_dataset_by_name(dataset_name, split)
            if loaded_dataset is not None:
                datasets.append(loaded_dataset)
                loaded_dataset_names.append(dataset_name)
            else:
                print(f"WARNING: Failed to load dataset: {dataset_name}")
        
        if not datasets:
            raise ValueError(f"No valid datasets found in: {dataset}")
        
        # Simplify datasets before concatenation
        simplified_datasets = [ds.select_columns(['audio', 'text']) for ds in datasets]
        
        # Concatenate datasets
        self.dataset = concatenate_datasets(simplified_datasets)
        original_size = len(self.dataset)
        
        # Process the combined dataset
        if self.config.dataset.do_filter:
            self._process_dataset(original_size)
        else:
            # Shuffle if not sorting by length
            self._shuffle_dataset()

    def _load_and_process_single_dataset(self, dataset, split):
        """Load and process a single dataset.
        
        Args:
            dataset: Dataset name
            split: Dataset split to load
        """
        self.dataset = self._load_dataset_by_name(dataset, split)
        if self.dataset is None:
            raise ValueError(f"Failed to load dataset: {dataset}")
        
        # Process the dataset
        if self.config.dataset.do_filter:
            original_size = len(self.dataset)
            self._process_dataset(original_size)
        else:
            # Shuffle if not sorting by length
            self._shuffle_dataset()

    def _process_dataset(self, original_size):
        """Process the dataset with filtering, combining speakers, and sorting.
        
        Args:
            original_size: Original size of the dataset before processing
        """
        # Filter by audio length and get audio lengths for later use
        self._filter_by_audio_length(original_size)
        
        # Combine speakers if enabled (for training split only)
        self._combine_speakers_if_enabled()
        
        # Sort by audio length or shuffle
        self._sort_or_shuffle_dataset()

    def _filter_by_audio_length(self, original_size):
        """Filter dataset by audio length and cache audio lengths.
        
        Args:
            original_size: Original size of the dataset before filtering
        """
        try:
            # Cache audio lengths and filter in a single pass
            all_lengths = []
            filtered_indices = []
            
            for i, example in enumerate(self.dataset):
                length = self._get_audio_length(example)
                if self.min_audio_length <= length <= self.max_audio_length:
                    filtered_indices.append(i)
                    all_lengths.append((i, length))
            
            # Store cached lengths for later use
            self.audio_lengths = dict(all_lengths)
            
            # Apply filter
            self.dataset = self.dataset.select(filtered_indices)
            print(f"Dataset with {len(self.dataset)} samples (filtered from {original_size} by audio length: {self.min_audio_length}s-{self.max_audio_length}s)")
        except Exception as e:
            # If caching fails, fall back to standard filter
            print(f"Warning: Could not optimize length filtering: {e}")
            self.dataset = self.dataset.filter(self._length_filter, num_proc=8)
            print(f"Dataset with {len(self.dataset)} samples (filtered from {original_size} by audio length: {self.min_audio_length}s-{self.max_audio_length}s)")

    def _length_filter(self, sample):
        """Filter function to keep only samples with appropriate audio lengths.
        
        Args:
            sample: Dataset sample to filter
        
        Returns:
            True if sample should be kept, False otherwise
        """
        try:
            audio = sample.get("audio", None)
            if audio is None:
                return True
                
            # Handle dictionary format (most common)
            if isinstance(audio, dict):
                if "array" in audio and "sampling_rate" in audio:
                    audio_array = audio["array"]
                    sampling_rate = audio["sampling_rate"]
                    audio_length = len(audio_array) / sampling_rate
                    return self.min_audio_length <= audio_length <= self.max_audio_length
            
            # Handle numpy array or tensor format
            elif hasattr(audio, "shape") or hasattr(audio, "size"):
                # Assume 16kHz sampling rate if not specified
                sampling_rate = 16000
                if hasattr(audio, "shape"):
                    audio_length = audio.shape[0] / sampling_rate
                else:
                    audio_length = audio.size(0) / sampling_rate
                return self.min_audio_length <= audio_length <= self.max_audio_length
                    
            return True  # Keep samples with unknown format
        except Exception:
            return True  # Keep samples with unexpected structure

    def _get_audio_length(self, example):
        """Helper function to get audio length from an example.
        
        Args:
            example: Dataset example
            
        Returns:
            Audio length in seconds or 0.0 if it cannot be determined
        """
        # First check if we've already cached this length
        if hasattr(self, 'audio_lengths') and id(example) in self.audio_lengths:
            return self.audio_lengths[id(example)]
            
        try:
            audio = example.get("audio", None)
            if audio is None:
                return 0.0
                
            # Handle dictionary format (most common)
            if isinstance(audio, dict):
                if "array" in audio and "sampling_rate" in audio:
                    audio_array = audio["array"]
                    sampling_rate = audio["sampling_rate"]
                    return len(audio_array) / sampling_rate
            
            # Handle numpy array or tensor format
            elif hasattr(audio, "shape") or hasattr(audio, "size"):
                # Assume 16kHz sampling rate if not specified
                sampling_rate = 16000
                if hasattr(audio, "shape"):
                    return audio.shape[0] / sampling_rate
                else:
                    return audio.size(0) / sampling_rate
                    
            return 0.0  # Default for unknown format
        except Exception:
            return 0.0  # Default for errors

    def _combine_speakers_if_enabled(self):
        """Combine speakers if enabled in config (for training split only)."""
        if (hasattr(self.config, 'dataset') and 
            hasattr(self.config.dataset, 'combine_speakers') and 
            self.config.dataset.combine_speakers and 
            "train" in self.split): 
            self._combine_speakers()

    def _combine_speakers(self):
        """Combine speakers in the dataset, replacing original samples."""
        # Check which samples are short enough to be combined
        half_max_length = self.max_audio_length / 2
        
        # Use cached lengths if available, otherwise calculate them
        if hasattr(self, 'audio_lengths'):
            samples_with_length = [(i, self.audio_lengths.get(id(self.dataset[i]), self._get_audio_length(self.dataset[i]))) 
                                 for i in range(len(self.dataset))]
        else:
            samples_with_length = [(i, self._get_audio_length(example)) 
                                 for i, example in enumerate(self.dataset)]
            
        short_samples = [(i, length) for i, length in samples_with_length if length <= half_max_length]
        
        print(f"Found {len(short_samples)} samples shorter than {half_max_length}s that are candidates for combination")
        
        # If we have short samples, combine them
        if len(short_samples) >= 2:
            # Shuffle to randomize combinations
            random.shuffle(short_samples)
            
            # Create pairs of samples to combine
            combined_pairs = []
            used_indices = set()
            
            for i in range(0, len(short_samples) - 1, 2):
                idx1, len1 = short_samples[i]
                idx2, len2 = short_samples[i + 1]
                
                # Skip if combined length exceeds max_audio_length
                if len1 + len2 > self.max_audio_length:
                    continue
                
                combined_pairs.append((idx1, idx2))
                used_indices.add(idx1)
                used_indices.add(idx2)
            
            print(f"Creating {len(combined_pairs)} combined samples")
            
            # Create combined samples
            combined_samples = self._create_combined_samples(combined_pairs)
            
            # Get indices of samples that weren't combined
            unused_indices = [i for i in range(len(self.dataset)) if i not in used_indices]
            
            # Create a new dataset with the unused and combined samples
            unused_samples = [self.dataset[i] for i in unused_indices]
            all_samples = unused_samples + combined_samples
            self.dataset = Dataset.from_list(all_samples)
            print(f"Created {len(combined_samples)} combined samples, keeping {len(unused_indices)} unused samples. New dataset size: {len(self.dataset)}")
            
            # Reset audio lengths cache
            if hasattr(self, 'audio_lengths'):
                delattr(self, 'audio_lengths')
            
            # Filter again to make sure all samples respect max_audio_length
            original_after_combine = len(self.dataset)
            self._filter_by_audio_length(original_after_combine)
            if len(self.dataset) < original_after_combine:
                print(f"Filtered out {original_after_combine - len(self.dataset)} samples that exceeded max audio length")

    def _create_combined_samples(self, combined_pairs):
        """Create combined samples from pairs.
        
        Args:
            combined_pairs: List of (index1, index2) tuples for pairs to combine
            
        Returns:
            List of combined samples
        """
        combined_samples = []
        
        for idx1, idx2 in combined_pairs:
            sample1 = self.dataset[idx1]
            sample2 = self.dataset[idx2]
            
            # Create a new sample that combines the audio and text of the two samples
            combined_sample = deepcopy(sample1)
            
            # Combine audio arrays
            audio1 = sample1["audio"]
            audio2 = sample2["audio"]
            
            if isinstance(audio1, dict) and "array" in audio1 and "sampling_rate" in audio1:
                # Concatenate audio arrays along time axis
                combined_audio = np.concatenate([audio1["array"], audio2["array"]])
                combined_sample["audio"]["array"] = combined_audio
            
            # Combine transcripts with speaker labels
            combined_sample["text"] = f"s1: {sample1['text']}. s2: {sample2['text']}"
            
            combined_samples.append(combined_sample)
        
        return combined_samples

    def _sort_or_shuffle_dataset(self):
        """Sort dataset by length or shuffle it based on config."""
        if (hasattr(self.config, 'dataset') and 
            hasattr(self.config.dataset, 'sort_by_length') and 
            self.config.dataset.sort_by_length):
            self._sort_by_length()
        else:
            self._shuffle_dataset()

    def _sort_by_length(self):
        """Sort dataset by audio length (longest first) using cached lengths if available."""
        if hasattr(self, 'audio_lengths') and len(self.audio_lengths) == len(self.dataset):
            # Use cached lengths if available
            indices = sorted(range(len(self.dataset)), 
                            key=lambda i: self.audio_lengths.get(id(self.dataset[i]), 0.0), 
                            reverse=True)
        else:
            # Calculate lengths if not cached
            lengths = [self._get_audio_length(example) for example in self.dataset]
            indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        
        self.dataset = self.dataset.select(indices)
        print("Dataset sorted by audio length (longest first)")

    def _shuffle_dataset(self):
        """Shuffle the dataset."""
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        self.dataset = self.dataset.select(indices)

    def _init_tokenizer_and_processor(self, lm_model, tokenizer, processor):
        """Initialize tokenizer, processor, and audio token.
        
        Args:
            lm_model: Language model name
            tokenizer: Tokenizer instance or None
            processor: Processor instance or None
        """
        # Initialize tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.tokenizer = tokenizer
        
        # Set up chat template
        self._setup_chat_template()
        
        # Initialize processor
        if processor is not None:
            self.processor = processor
        elif (self.config is not None and 
            hasattr(self.config, 'model') and 
            hasattr(self.config.model, 'encoder_model_name') and 
            "mimi" in self.config.model.encoder_model_name.lower()):
            self.processor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        else:
            self.processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
        
        # Set up audio token
        self.audio_token = "<|audio|>"
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
    
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
            except Exception:
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
            if (len(audio_array) / audio_sampling_rate) > 16: # truncate audio to 16 seconds
                audio_array = audio_array[:audio_sampling_rate*16]
            return audio_array
        elif "MimiFeatureExtractor" in self.processor.__class__.__name__ or "Mimi" in self.processor.__class__.__name__:
            # Process with Mimi feature extractor following the HuggingFace example
            return self.processor(
                raw_audio=audio_array,
                sampling_rate=target_sampling_rate,
                return_tensors="pt"
            )
        elif ("SeamlessM4TFeatureExtractor" in self.processor.__class__.__name__):
            if (len(audio_array) / audio_sampling_rate) > 16: # truncate audio to 16 seconds
                audio_array = audio_array[:audio_sampling_rate*16]
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
            return self.processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True, output_hidden_states=True)
    
    def _phonemize(self, text, language='en-us'):
        ipa_text = phonemize(
            text,
            language='en-us',
            backend='espeak',
            words_mismatch="ignore",
            strip=True,
            preserve_punctuation=True,
            with_stress=False,
            logger=None,)
        return ipa_text

    def _format_chat(self, example, audio):
        prompt = "Transcribe the following text in lower case:"

        if self.config.training.do_phonemize:
            example = self._phonemize(example)
            prompt = "Transcribe the following text in IPA:"

        if self.lm_model_name != "gemma":
            return [
                {"role": "system", "content": "You transcribe speech to text. You need to repeat the text once exactly as it is spoken."},
                {"role": "user", "content": f"{prompt} {audio}"},
                {"role": "assistant", "content": example}
            ]
        else:
            return [
                {"role": "user", "content": f"{prompt} {audio}"},
                {"role": "assistant", "content": example}
            ]
        
    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio = example["audio"]
        text = example["text"].lower()
        # Handle potential missing fields in simplified datasets
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            audio_array = torch.tensor(audio["array"].copy(), dtype=torch.float32) # copy to circumvent stride issues (torch can't handle negative strides for some reason)
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
            "raw_audio": audio_array,
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


def collate_fn(batch, tokenizer, T, include_raw_audio: bool = False):
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
        "raw_audio": [item["raw_audio"] for item in batch] if include_raw_audio else None
    } 

def add_gaussian_noise(
    item: dict,
    p: float=0.2,
    low: float=0.001,
    high:float=0.03
) -> dict:
    """Apply a uniformly sampled noise to the raw audio waveform amplitudes.
    
    Sample same length as audio --> multiply with audio amplitudes --> add to audio and return
    """

    # Return original waveform if augmentation is skipped
    if np.random.random() > p:
        return item

    audio: np.ndarray = item["audio"][0]["array"]
    noise = np.random.uniform(low, high, audio.shape)
    item["audio"][0]["array"] += noise * audio
    return item


def apply_band_stop_augmentation(
    item: dict,
    p: float=0.2,
    min_freq: int=0,
    max_freq: int=4000,
    min_width: int=100
) -> dict:
    """
    Apply a band stop filter as a data augmentation technique for ASR.
    Inspired by: WHISPER TURNS STRONGER: AUGMENTING WAV2VEC 2.0 
                 FOR SUPERIOR ASR IN LOW-RESOURCE LANGUAGES

    min_freq/max_freq: limits the frequencies to apply the bandwith filter to.
    min_width: minimum width of the stop band.
    p: probability of applying the augmentation.
    fs: sampling rate of the audio.
    """
    #  Return original waveform if augmentation is skipped
    if np.random.random() > p:
        return item

    waveform = item["audio"][0]["array"]
    sr = item["audio"][0]["sampling_rate"]

    # Define the frequency range for the stop band
    max_start_freq = max_freq - min_width  # Ensure room for the band width
    f_low = np.random.uniform(min_freq, max_start_freq)  # Random start of stop band
    f_high = np.random.uniform(f_low + min_width, min(f_low + 200, max_freq))  # Random end, capped at max_freq

    # Design a Butterworth band stop filter (order 4 for moderate roll-off)
    sos = butter(4, [f_low, f_high], btype='bandstop', output='sos', fs=sr)

    # Apply zero-phase filtering
    augmented_waveform = sosfiltfilt(sos, waveform)
    item["audio"][0]["array"] = augmented_waveform

    return item


def apply_pitch_shift_augmentation(
    item: dict,
    p: float=0.2,
    min_shift: int=-6,
    max_shift: int=6
) -> dict:
    """
    Apply pitch shift augmentation to an audio waveform using librosa.
    Inspired by: WHISPER TURNS STRONGER: AUGMENTING WAV2VEC 2.0 
                 FOR SUPERIOR ASR IN LOW-RESOURCE LANGUAGES
    
    p: probability of applying the augmentation.
    min_shift: minimum pitch shift in semitones.
    max_shift: maximum pitch shift in semitones.
    """
    # Return original waveform if augmentation is skipped
    if np.random.random() > p:
        return item

    waveform = item["audio"][0]["array"]
    sr = item["audio"][0]["sampling_rate"]

    # Randomly choose pitch shift in semitones between min_shift and max_shift
    n_steps = np.random.uniform(min_shift, max_shift)

    # Apply pitch shift using librosa
    augmented_waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
    item["audio"][0]["array"] = augmented_waveform

    return item


def filter_difficult_examples(example):
    text = example["text"]
    
    # 1. Filter out spelled out uppercase letters (D A N S E H A L L E R)
    if re.search(r'(^|\s)[A-Z](\s+[A-Z]){2,}($|\s|[,.!?])', text):
        return False
    
    # 2. Filter out sentences with backslashes
    if "\\" in text:
        return False
    
    lower_text = text.lower()
    words = re.findall(r'\b\w+\b', lower_text)

    # 3. Filter out sentences with repeated adjacent words
    for i in range(len(words) - 1):
        if words[i] == words[i+1] and len(words[i]) > 1:  # Ignore single letters
            return False
    
    return True  # Keep this example

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