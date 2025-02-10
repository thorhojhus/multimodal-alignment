import torch
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
        if dataset == "librispeech":
            from dataset.load_librispeech import librispeech_dataset_dict
            self.dataset = librispeech_dataset_dict[str(split)]
        else:
            print("Unsupported dataset")
        self.tokenizer = tokenizer
        self._setup_chat_template()
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

    def _format_chat(self, example, audio):
        if self.lm_model_name != "gemma":
            return [
                {"role": "system", "content": f"You transcribe speech to text. You need to repeat the text exactly as it is spoken."},
                {"role": "user", "content": f"Transcribe and repeat the following text in lower case: {audio}"},
                {"role": "assistant", "content": example}
            ]
        else:
            return [
                {"role": "user", "content": f"Transcribe and repeat the following text in lower case: {audio}"},
                {"role": "assistant", "content": example}
            ]
        
    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio = example["audio"]
        text = example["text"].lower()
        audio_features = self._audio_to_features(audio)

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

        assistant_response = self.tokenizer.apply_chat_template(
            [conversation[-1]],
            add_generation_prompt=False,
            return_tensors="pt"
        )[0]
        
        labels = torch.full(tokenized.size(), fill_value=-100)
        labels[-len(assistant_response):] = assistant_response

        alt_labels = torch.full(tokenized_alt.size(), fill_value=-100)
        alt_labels[-len(assistant_response):] = assistant_response
        
        return {
            "input_ids": tokenized,
            "audio_features": audio_features,
            "labels": labels,
            "alt_input_ids": tokenized_alt,
            "alt_labels": alt_labels
        }

def collate_fn(batch):
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
    


    return {
        "input_ids": torch.stack(input_ids),
        "audio_features": torch.stack(audio_features),
        "labels": torch.stack(labels),
        "alt_input_ids": torch.stack(alt_input_ids),
        "alt_labels": torch.stack(alt_labels)
    }

if __name__ == "__main__":
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    dataset = AudioChatDataset(tokenizer=tokenizer, processor=processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break