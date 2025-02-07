import torch
from transformers import AutoProcessor
from datasets import load_dataset
from dataset.load_librispeech import librispeech_dataset_dict
from dataset.load_coral import coral_dataset_dict
from truecase import get_true_case

class AudioChatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset="coral", split="train.clean.100", tokenizer=None, processor=None):
        if dataset == "coral":
            self.dataset = coral_dataset_dict[split.split(".")[0]]
        if dataset == "librispeech":
            self.dataset = librispeech_dataset_dict[str(split)]
        else:
            print("Unsupported dataset")
        self.tokenizer = tokenizer
        self.processor = processor or AutoProcessor.from_pretrained("openai/whisper-base.en")
        self.audio_token = "<|audio|>"
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)

    ### NOTE: this is a pretty gnarly way to do it
    ### Should probably be done in another way;
    try:
        get_true_case("Any lookup erros?")
    except LookupError:
        import nltk
        nltk.download("punkt_tab")
        get_true_case("Any lookup erros?")

    def __len__(self):
        return len(self.dataset)

    def _audio_to_features(self, audio):
        audio = audio["array"]

        inputs = self.processor(
            audio, 
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            truncation=True,
            #max_length=audio.size
        )
        return inputs.input_features[0]

    def _format_chat(self, example):
        return [
            {"role": "user", "content": f"Repeat the following: {self.audio_token}"},
            {"role": "assistant", "content": get_true_case(example)}
        ]

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio = example["audio"]
        text = example["text"].lower().strip()
        audio_features = self._audio_to_features(audio)

        # format as chat
        conversation = self._format_chat(text)
        
        # tokenize conversation with special handling for audio token
        tokenized = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            return_tensors="pt"
        )[0]

        # find audio token position and replace with actual token ID
        audio_token_idx = (tokenized == self.audio_token_id).nonzero()

        tokenized[audio_token_idx] = self.audio_token_id
        
        return {
            "input_ids": tokenized,
            "audio_features": audio_features,
            "labels": self._create_labels(tokenized, conversation)
        }

    def _create_labels(self, tokenized, conversation):
        # mask all tokens except assistant response
        labels = tokenized.clone()
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        # find all BOS and EOS positions
        bos_positions = (tokenized == bos_token_id).nonzero(as_tuple=True)[0]
        eos_positions = (tokenized == eos_token_id).nonzero(as_tuple=True)[0]

        # get the last BOS token (start of assistant response)
        last_bos = bos_positions[-1].item()
        response_start = last_bos

        # find the first EOS after last BOS (end of assistant response)
        eos_after_bos = eos_positions[eos_positions > last_bos]
        response_end = eos_after_bos[0].item() if len(eos_after_bos) > 0 else len(tokenized)

        # mask non-response parts
        labels[:response_start] = -100
        labels[response_end:] = -100
        return labels

def collate_fn(batch):
    # handle text inputs
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    audio_features = []
    
    for item in batch:
        # pad input_ids and labels
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=0))
        labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        audio_features.append(item["audio_features"])

    return {
        "input_ids": torch.stack(input_ids),
        "audio_features": torch.stack(audio_features),
        "labels": torch.stack(labels)
    }

if __name__ == "__main__":
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    dataset = LibriSpeechChatDataset(tokenizer=tokenizer, processor=processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break