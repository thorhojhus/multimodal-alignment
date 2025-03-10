from torch.nn import functional as F
import torch.nn as nn
import transformers
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from dataset.load_librispeech import librispeech_dataset_dict
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.bfloat16)

model_name = "google/gemma-2-2b-it"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
quant_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
llm = AutoLigerKernelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=quant_config)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
speecht5 = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts",
                                                torch_dtype=torch.bfloat16,
                                                device_map="cuda",)
speecht5_encoder = speecht5.get_encoder().wrapped_encoder
speecht5_decoder = speecht5.get_decoder().wrapped_decoder
speecht5_prenet = speecht5.get_decoder().prenet
post_net = speecht5.speech_decoder_postnet

class LLMToSpeechT5Adapter(nn.Module):
    def __init__(self, llm_dim=2304, target_dim=768, upscale_factor=10):
        super().__init__()
        self.projection = nn.Linear(llm_dim, target_dim)
        self.upscale_factor = upscale_factor
        self.upscale = nn.Linear(1, upscale_factor)
        self.norm = nn.LayerNorm(target_dim)
        
        # For processing paired hidden states 
        self.paired_projection = nn.Linear(llm_dim, target_dim)
        self.paired_norm = nn.LayerNorm(target_dim)
        
    def process_single_state(self, hidden_states, projection, norm_layer):
        B, T, D = hidden_states.shape
        # Project to target dimension
        x = projection(hidden_states)  # [B, T, target_dim]
        # Reshape for temporal upscaling
        x = x.transpose(1, 2)  # [B, target_dim, T]
        # Upscale the sequence length
        x = x.unsqueeze(3)  # [B, target_dim, T, 1]
        x = self.upscale(x)  # [B, target_dim, T, upscale_factor]
        x = x.flatten(2)  # [B, target_dim, T*upscale_factor]
        x = x.transpose(1, 2)  # [B, T*upscale_factor, target_dim]
        # Normalize
        x = norm_layer(x)
        return x
        
    def forward(self, last_hidden_state, paired_hidden_states=None):
        # Process last hidden state
        adapted_last = self.process_single_state(
            last_hidden_state, 
            self.projection, 
            self.norm
        )
        
        # Process paired hidden states if provided
        if paired_hidden_states is not None:
            # Get dimensions
            num_pairs, B, T, D = paired_hidden_states.shape
            
            # Process each pair and collect results
            adapted_paired = []
            for i in range(num_pairs):
                pair_state = paired_hidden_states[i]  # [B, T, D]
                processed = self.process_single_state(
                    pair_state,
                    self.paired_projection,
                    self.paired_norm
                )
                adapted_paired.append(processed)
            
            # Stack back to original format
            adapted_paired = torch.stack(adapted_paired)  # [num_pairs, B, T*upscale, target_dim]
            
            return adapted_last, adapted_paired
        
        return adapted_last

# Freeze the LLM
for param in llm.parameters():
    param.requires_grad = False

for param in speecht5_decoder.parameters():
    param.requires_grad = False

for param in speecht5_prenet.parameters():
    param.requires_grad = False

for param in post_net.parameters():
    param.requires_grad = False

class LLMToSpeech(nn.Module):
    def __init__(self, llm, adapter, decoder, postnet):
        super().__init__()
        self.llm = llm
        self.adapter = adapter
        self.decoder = decoder
        self.postnet = postnet
        
    def forward(self, input_ids):
        # Get LLM outputs
        with torch.no_grad():
            llm_outputs = self.llm(input_ids, output_hidden_states=True)
            last_hidden_state = llm_outputs.hidden_states[-1]
            hidden_states = llm_outputs.hidden_states[:-1]
            
            # Stack hidden states to [num_layers, B, T, LLM_DIM]
            stacked_hidden_states = torch.stack(hidden_states)
            
            # Take mean of pairs of hidden states
            num_layers = stacked_hidden_states.shape[0]
            if num_layers % 2 == 0:  # Even number of layers
                reshaped = stacked_hidden_states.view(num_layers//2, 2, *stacked_hidden_states.shape[1:])
                paired_hidden_states = torch.mean(reshaped, dim=1)
            else:  # Odd number of layers
                reshaped = stacked_hidden_states[:-1].view((num_layers-1)//2, 2, *stacked_hidden_states.shape[1:])
                paired_hidden_states = torch.mean(reshaped, dim=1)
                paired_hidden_states = torch.cat([
                    paired_hidden_states, 
                    stacked_hidden_states[-1:] 
                ], dim=0)
        
        # Process both types of hidden states through the adapter
        adapted_last, adapted_paired = self.adapter(last_hidden_state, paired_hidden_states)
        
        decoder_outputs = self.decoder(adapted_last, encoder_hidden_states=adapted_paired)
        
        # Generate spectrograms
        _, outputs_after, _ = self.postnet(decoder_outputs.last_hidden_state)
        
        return outputs_after
    
adapter = LLMToSpeechT5Adapter().to("cuda", torch.bfloat16)
llm_to_speech = LLMToSpeech(llm, adapter, speecht5_decoder, post_net).to("cuda", torch.bfloat16)

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokens = tokenizer.encode(item["text"].lower())
        tokens = torch.tensor(tokens)
        return {"audio": item["audio"], "tokens": tokens}

ds = LibriSpeechDataset(librispeech_dataset_dict["train.clean.360"])

def collate_fn(batch):
    tokens = [item["tokens"] for item in batch]
    max_len = max(len(t) for t in tokens)
    
    # Create padded tensors and attention masks
    padded_tokens = torch.zeros(len(tokens), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(tokens), max_len, dtype=torch.long)
    
    for i, t in enumerate(tokens):
        padded_tokens[i, :len(t)] = t
        attention_mask[i, :len(t)] = 1
    
    audios = [torch.tensor(item["audio"]["array"]) for item in batch]
    audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    spectograms = torch.stack([torch.tensor(processor(audio_target=audio.cpu().flatten(), sampling_rate=16000)["input_values"]) for audio in audios]).squeeze(1)
    
    return {
        "audio": audios, 
        "tokens": padded_tokens, 
        "attention_mask": attention_mask,
        "spectograms": spectograms
    }


dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=8, prefetch_factor=8)

# Training loop
optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)

llm_to_speech.train()

for batch in dl:
    tokens = batch["tokens"].to("cuda")
    target_specs = batch["spectograms"].to("cuda")
    
    # Forward pass
    optimizer.zero_grad()
    outputs = llm_to_speech(tokens)
    
    # Calculate loss
    min_len = min(outputs.size(1), target_specs.size(1))
    loss = F.l1_loss(outputs[:, :min_len], target_specs[:, :min_len])
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")