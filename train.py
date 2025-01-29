from datasets import load_dataset, Audio
from transformers import AutoProcessor
import torch
from torch.utils.data import DataLoader
from models.audio_llm import AudioLLMModel, AudioLLMConfig, Projector 
import aiohttp

class LibriSpeechChatDataset(torch.utils.data.Dataset):
    def __init__(self, split="train.clean.100", tokenizer=None, processor=None):
        self.dataset = load_dataset("librispeech_asr", split=split, trust_remote_code=True, storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
        self.tokenizer = tokenizer
        self.processor = processor or AutoProcessor.from_pretrained("openai/whisper-base.en")
        self.audio_token = "<|audio|>"
        
    def __len__(self):
        return len(self.dataset)

    def _audio_to_features(self, audio):
        audio = audio["array"]
        inputs = self.processor(
            audio, 
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            truncation=True
        )
        return inputs.input_features[0]

    def _format_chat(self, example):
        return [
            {"role": "user", "content": f"Transcribe {self.audio_token}"},
            {"role": "assistant", "content": example["text"]}
        ]

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio_features = self._audio_to_features(example["audio"])
        
        # format as chat
        conversation = self._format_chat(example)
        
        # tokenize conversation with special handling for audio token
        tokenized = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        )[0]
        
        # find audio token position and replace with actual token ID
        audio_token_idx = (tokenized == self.tokenizer.convert_tokens_to_ids("{{content}}")).nonzero()
        tokenized[audio_token_idx] = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        
        return {
            "input_ids": tokenized,
            "audio_features": audio_features,
            "labels": self._create_labels(tokenized, conversation)
        }

    def _create_labels(self, tokenized, conversation):
        # mask all tokens except assistant response
        labels = tokenized.clone()
        sep = self.tokenizer.eos_token_id
        response_start = None
        
        # find the start of the assistant response
        for i, token in enumerate(labels):
            if token == sep:
                if response_start is None:
                    response_start = i + 1
                else:
                    response_end = i
                    break
        
        # mask non-response parts
        labels[:response_start] = -100
        labels[response_end:] = -100
        return labels

def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    audio_features = []
    
    for item in batch:
        # pad input_ids and labels
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=0))
        labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        
        # audio features
        audio_features.append(item["audio_features"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "audio_features": torch.stack(audio_features),
        "labels": torch.stack(labels)
    }

def train(model, dataset, num_epochs=3, batch_size=4):
    # create optimizer only for trainable parameters
    trainable_params = [
        {"params": model.lm.get_input_embeddings().parameters()},
        {"params": model.projector.parameters()}
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # move data to GPU
            input_ids = batch["input_ids"].to("cuda")
            audio_features = batch["audio_features"].to("cuda", torch.bfloat16)
            labels = batch["labels"].to("cuda")
            
            # forward pass
            outputs = model(
                input_ids=input_ids,
                audio_features=audio_features,
                labels=labels
            )
            
            # backward pass
            loss = outputs.loss
            loss.backward()
            
            # optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # initialize components
    config = AudioLLMConfig()
    model = AudioLLMModel(config)
    
    # create dataset and processor
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    dataset = LibriSpeechChatDataset(
        split="train.clean.100",
        tokenizer=model.tokenizer,
        processor=processor
    )
    
    # freeze non-trainable components
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.lm.parameters():
        param.requires_grad = False
    model.lm.get_input_embeddings().weight.requires_grad = True
    
    # start training
    train(model, dataset, num_epochs=3, batch_size=4)