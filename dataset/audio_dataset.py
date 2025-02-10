import torch
from torch import Tensor
from transformers import AutoProcessor
from datasets import load_dataset
from truecase import get_true_case
from dataset.chat_templates import CHATML_TEMPLATE, LLAMA3_TEMPLATE, GEMMA_2

class AudioChatDataset(torch.utils.data.Dataset):
    def __init__(self, lm_model: str, dataset="coral", split="train.clean.100", tokenizer=None, processor=None):
        self.lm_model_name = lm_model.split("/")[-1].split("-")[0].lower()

        if dataset == "coral":
            from dataset.load_coral import coral_dataset_dict
            self.dataset = coral_dataset_dict[split.split(".")[0]]
        elif dataset == "librispeech":
            from dataset.load_librispeech import librispeech_dataset_dict
            self.dataset = librispeech_dataset_dict[str(split)]
        else:
            print("Unsupported dataset")
        self.tokenizer = tokenizer
        self._setup_chat_template()
        self.processor = processor or AutoProcessor.from_pretrained("openai/whisper-base.en")
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
        if self.lm_model_name != "gemma":
            return [
                {"role": "system", "content": f"You transcribe speech to text. You need to repeat the text exactly as it is spoken."},
                {"role": "user", "content": f"Transcribe and repeat the following text in lower case: {self.audio_token}"},
                {"role": "assistant", "content": example.lower()}
            ]
        else:
            return [
                {"role": "user", "content": f"Transcribe and repeat the following text in lower case: {self.audio_token}"},
                {"role": "assistant", "content": example.lower()}
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

        assistant_response = self.tokenizer.apply_chat_template(
            [conversation[-1]],
            add_generation_prompt=False,
            return_tensors="pt"
        )[0]
        
        labels = torch.full(tokenized.size(), fill_value=-100)
        labels[-len(assistant_response):] = assistant_response

        return {
            "input_ids": tokenized,
            "audio_features": audio_features,
            "labels": labels,
            "tokenizer": self.tokenizer, #NOTE: expose internal tokenizer to collate_fn below
        }


def get_audio_positions(input_ids: Tensor, tokenizer) -> list[Tensor]:
    # find positions of audio tokens in each sequence
    return [(row == tokenizer.convert_tokens_to_ids("<|audio|>")).nonzero(as_tuple=True)[0] for row in input_ids]


def adjust_insertion_indices(audio_positions: list[Tensor]) -> list[Tensor]:
    # adjust indices for audio token insertion
    return [
        pos - torch.arange(len(pos), device=pos.device) if pos.numel() > 0
        else torch.tensor([], dtype=torch.long, device=pos.device)
        for pos in audio_positions
    ]


def get_valid_positions(input_ids: Tensor, tokenizer) -> list[Tensor]:
    # get positions of non-audio tokens
    return [(row != tokenizer.convert_tokens_to_ids("<|audio|>")).nonzero(as_tuple=True)[0] for row in input_ids]


def prepare_inputs(
    input_ids: Tensor, labels: Tensor | None, tokenizer
) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
    # prepare input tensors for model
    attention_mask = torch.ones_like(input_ids)  # NOTE: This does mask out padding correctly right now.
    if labels is None:
        labels = input_ids.clone()

    audio_positions = get_audio_positions(input_ids, tokenizer)
    insertion_indices = adjust_insertion_indices(audio_positions)
    valid_positions = get_valid_positions(input_ids, tokenizer)

    # remove audio tokens and pad sequences
    new_input_ids = [input_ids[i][pos] for i, pos in enumerate(valid_positions)]
    new_labels = [labels[i][pos] for i, pos in enumerate(valid_positions)]
    new_attention_mask = [attention_mask[i][pos] for i, pos in enumerate(valid_positions)]

    new_input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=-100)
    new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
    
    return new_input_ids, new_attention_mask, new_labels, insertion_indices


def collate_fn(batch):
    # handle text inputs
    max_len = max(len(item["input_ids"]) for item in batch)
    tokenizer = batch[-1]["tokenizer"]
    input_ids = []
    labels = []
    audio_features = []

    for item in batch:
        # pad input_ids and labels
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=0))
        labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        audio_features.append(item["audio_features"])

    input_ids = torch.stack(input_ids)
    audio_features = torch.stack(audio_features)
    labels = torch.stack(labels)

    prepared_input_ids, prepared_attention_mask, prepared_labels, prepared_insertion_indices = prepare_inputs(input_ids, labels, tokenizer)
    
    return {
        "input_ids": prepared_input_ids,
        "audio_features": audio_features,
        "labels": prepared_labels,
        "attention_mask": prepared_attention_mask,
        "insertion_indices": prepared_insertion_indices
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