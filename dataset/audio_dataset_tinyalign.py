import torch
import torchaudio
from transformers import AutoProcessor, AutoTokenizer
from transformers import BatchFeature

"""

A TinyAlign specific Dataset Class. Implements the neccesary preprocessing steps required for TinyAlign Bridgeformer.

"""

class AudioChatDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 lm_model: str="HuggingFaceTB/SmolLM2-360M-Instruct",
                 dataset: str="coral", split="train.clean.100", 
                 tokenizer=None, 
                 processor=None,
                 T=30):
        self.lm_model_name = lm_model.split("/")[-1].split("-")[0].lower()
        if dataset == "coral":
            from dataset.load_coral import coral_dataset_dict
            self.dataset = coral_dataset_dict[split.split(".")[0]]
        elif dataset == "librispeech":
            from dataset.load_librispeech import librispeech_dataset_dict
            self.dataset = librispeech_dataset_dict[str(split)]
        else:
            print("Unsupported dataset")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.tokenizer = tokenizer
        self.T = T
        self.processor = processor or AutoProcessor.from_pretrained("openai/whisper-base.en")
        print(self.processor.__class__.__name__)
        self.audio_token = "<|audio|>"
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)

    def __len__(self):
        return len(self.dataset)

    def _audio_to_features(self, audio):
        audio_array = torch.tensor(audio["array"], dtype=torch.float32)
        audio_sampling_rate = audio["sampling_rate"]
        if "SeamlessM4TFeatureExtractor" in self.processor.__class__.__name__:
            target_sampling_rate = 16000
        else:
            target_sampling_rate = self.processor.feature_extractor.sampling_rate

        # resample if necessary
        if audio_sampling_rate != target_sampling_rate:
            audio_array = torchaudio.functional.resample(
                audio_array, 
                orig_freq=audio_sampling_rate, 
                new_freq=target_sampling_rate
            )

        if ("Wav2Vec2" in self.processor.__class__.__name__):
            # for wav2vec, return raw waveform
            return audio_array
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

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio = example["audio"]
        text = example["text"].lower()
        audio_features = self._audio_to_features(audio)
        
        return {
            "input_ids": torch.tensor(self.tokenizer.encode(text)),
            "audio_features": audio_features,
        }

def collate_fn(batch, T: int):    
    pad_token_id = 0
    input_ids = []
    audio_features = []
    
    for item in batch:
        # pad input_ids and labels
        if len(item["input_ids"]) > T:
            item["input_ids"] = item["input_ids"][:T] # truncate as described in TinyAlign
        else:
            pad_len = T - len(item["input_ids"])      # or pad as described in TinyAlign
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        audio_features.append(item["audio_features"])

    input_ids = torch.stack(input_ids)

    # handle audio features based on model type
    if isinstance(audio_features[0], torch.Tensor):
        # for wav2vec and other raw waveform models
        audio_features = torch.nn.utils.rnn.pad_sequence(
            audio_features, 
            batch_first=True, 
            padding_value=0.0
        )
    elif isinstance(audio_features[0], BatchFeature):
        # for wav2vec2-bert and seamless models
        # Extract and pad the features
        features_list = [af.input_features[0] for af in audio_features]  # Remove batch dimension
        audio_features = torch.nn.utils.rnn.pad_sequence(
            features_list,
            batch_first=True,
            padding_value=0.0
        )
            
        audio_features = {
            'input_features': audio_features,
            'attention_mask': None
        }
    else:
        # for spectrogram-based models (Whisper, CLAP)
        audio_features = torch.stack(audio_features)
    
    return {
        "audio_features": audio_features,
        "input_ids": input_ids
    }
