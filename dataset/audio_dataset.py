import torch
import torchaudio
from torch import Tensor
from transformers import AutoProcessor, AutoTokenizer
from transformers import BatchFeature

from dataset.chat_templates import CHATML_TEMPLATE, GEMMA_2, LLAMA3_TEMPLATE


class AudioChatDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 lm_model: str="HuggingFaceTB/SmolLM2-360M-Instruct",
                 dataset: str="coral", split="train.clean.100", 
                 tokenizer=None, 
                 processor=None,
                 do_qa:bool=True):
        self.lm_model_name = lm_model.split("/")[-1].split("-")[0].lower()
        self.dataset_name = dataset
        self.do_qa = do_qa
        if self.dataset_name == "coral":
            from dataset.load_coral import coral_dataset_dict
            self.dataset = coral_dataset_dict[split.split(".")[0]]
        elif self.dataset_name == "librispeech":
            from dataset.load_librispeech import librispeech_dataset_dict
            self.dataset = librispeech_dataset_dict[str(split)]
        else:
            print("Unsupported dataset")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.tokenizer = tokenizer
        self._setup_chat_template()
        self.processor = processor or AutoProcessor.from_pretrained("openai/whisper-base.en")
        print(self.processor.__class__.__name__)
        self.audio_token = "<|audio|>"
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)

    def __len__(self):
        return len(self.dataset)
    
    def _setup_chat_template(self):
        if self.lm_model_name == "smollm2":
            self.tokenizer.chat_template = CHATML_TEMPLATE
        elif self.lm_model_name == "llama":
            self.tokenizer.chat_template = LLAMA3_TEMPLATE
        elif self.lm_model_name == "gemma":
            self.tokenizer.chat_template = GEMMA_2
        else:
            print("Custom chat template not found")
            pass

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

    def _format_chat_qa(self, example, audio, gender, dialect, age):
        messages = []
        if self.lm_model_name != "gemma":
            messages.append({
                "role": "system", 
                "content": "You transcribe speech to text and answer questions. You need to transcribe the speech as it is spoken and answer the questions."
            })
        messages.extend([
            {"role": "user", "content": f"Listen to the following audio and transcribe it in lower case: {audio}. Based on what you hear, what is the gender of the speaker, what is their approximate age, and what is their dialect?"},
            {"role": "assistant", "content": f"Transcription: {example}. \nThe gender of the speaker is {gender}, they're in their {round(float(age) / 10) * 10}'ies and the dialect is {dialect}."}])
        return messages

        
    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio = example["audio"]
        text = example["text"].lower()
        audio_features = self._audio_to_features(audio)

        # format as chat
        if (self.dataset_name == "coral") and self.do_qa:
            conversation = self._format_chat_qa(text, self.audio_token, example["gender"], example["dialect"], example["age"])
            conversation_alt = self._format_chat_qa(text, text, example["gender"], example["dialect"], example["age"])
        else:
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


def collate_fn(batch, tokenizer):
    # handle text inputs
    max_len = max(len(item["input_ids"]) for item in batch)
    max_len_pad = max(len(item["alt_input_ids"]) for item in batch)
    pad_token_id = 0
    input_ids = []
    labels = []
    audio_features = []
    alt_input_ids = []
    alt_labels = []
    
    for item in batch:
        # pad input_ids and labels
        pad_len = max_len - len(item["input_ids"])
        alt_pad_len = max_len_pad - len(item["alt_input_ids"])
        
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        audio_features.append(item["audio_features"])
        
        alt_input_ids.append(torch.nn.functional.pad(item["alt_input_ids"], (0, alt_pad_len), value=pad_token_id))
        alt_labels.append(torch.nn.functional.pad(item["alt_labels"], (0, alt_pad_len), value=-100))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    alt_input_ids = torch.stack(alt_input_ids)
    alt_labels = torch.stack(alt_labels)

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
        
        # Create appropriate attention mask
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
        "alt_labels": alt_labels
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