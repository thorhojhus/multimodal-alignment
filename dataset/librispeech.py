import torch
from transformers import AutoProcessor
from datasets import load_dataset
import aiohttp # needed to get around timeout bug in librispeech_asr
from truecase import get_true_case     

class LibriSpeechChatDataset(torch.utils.data.Dataset):
    def __init__(self, split="train.clean.100", tokenizer=None, processor=None, n_examples=5):
        self.dataset = load_dataset("librispeech_asr", split=split, trust_remote_code=True, storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
        self.tokenizer = tokenizer
        self.processor = processor or AutoProcessor.from_pretrained("openai/whisper-base.en")
        self.audio_token = "<|audio|>"
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.n_examples = n_examples


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
        )
        return inputs.input_features[0]

    def _format_chat(self, examples):
        return [
            message
            for example in examples
            for message in [
                {"role": "user", "content": f"Repeat the following: {self.audio_token}"},
                {"role": "assistant", "content": get_true_case(example)}
            ]
        ]

    def __getitem__(self, idx):
        # calculate start and end indices for the examples
        start_idx = idx * self.n_examples
        end_idx = min(start_idx + self.n_examples, len(self.dataset))
        
        # get multiple examples
        examples = [self.dataset[i] for i in range(start_idx, end_idx)]
        
        # process audio from first example only 
        audio_samples = [example["audio"] for example in examples]
        audio_features = [self._audio_to_features(audio) for audio in audio_samples]
        audio_features = torch.stack(audio_features)
        
        # get texts from all examples
        texts = [example["text"].lower().strip() for example in examples]

        # format as chat
        conversation = self._format_chat(texts)
        # rest remains the same
        tokenized = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            return_tensors="pt"
        )[0]

        audio_token_idx = (tokenized == self.audio_token_id).nonzero()
        tokenized[audio_token_idx] = self.audio_token_id
        
        return {
            "input_ids": tokenized,
            "audio_features": audio_features,
            "labels": self._create_labels(tokenized, conversation)
        }
    
    def _create_labels(self, tokenized, conversation):
        assistant_responses = [message for message in conversation if message["role"] == "assistant"]
        assistant_responses = [self.tokenizer.apply_chat_template([response], add_generation_prompt=False, return_tensors="pt")[0] for response in assistant_responses]
        assistant_responses_list = []
        for response in assistant_responses:
            bos_tokens = (response == self.tokenizer.bos_token_id).nonzero(as_tuple=True)[0]
            # split at the last BOS token
            last_bos = bos_tokens[-1].item()
            response = response[last_bos:]
            assistant_responses_list.append(response)

        # Start with all tokens masked
        labels = tokenized.clone()
        labels.fill_(-100)
        
        # Find and unmask the assistant responses in the full conversation
        current_pos = 0
        for response in assistant_responses_list:
            # For each response, find where it starts in the full conversation
            response_len = len(response)
            found = False
            
            # Search for the response starting from the current position
            while current_pos <= len(tokenized) - response_len and not found:
                if torch.equal(tokenized[current_pos:current_pos + response_len], response):
                    # When found, unmask these positions by copying the token ids
                    labels[current_pos:current_pos + response_len] = tokenized[current_pos:current_pos + response_len]
                    current_pos += response_len
                    found = True
                else:
                    current_pos += 1
        return labels
    
    def __len__(self):
        return len(self.dataset) // self.n_examples

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