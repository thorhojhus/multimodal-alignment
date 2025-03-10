import torch
import torch.nn as nn
import transformers
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from torch.nn import functional as F
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
from torch.utils.data import DataLoader

# Fix the tokenizer warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataset.load_librispeech import librispeech_dataset_dict

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

model_name = "google/gemma-2-2b-it"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
quant_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
llm = AutoLigerKernelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=quant_config)
tokens = tokenizer.encode("Hello world", return_tensors="pt")

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
speecht5 = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts",
                                                torch_dtype=torch.bfloat16,
                                                device_map="cuda",)
speecht5_encoder = speecht5.get_encoder().wrapped_encoder
speecht5_decoder = speecht5.get_decoder().wrapped_decoder
speecht5_prenet = speecht5.get_decoder().prenet
post_net = speecht5.speech_decoder_postnet


B, T = tokens.shape
TARGET_DIM = 768
upscale_factor = 4
hidden_states = llm.forward(tokens, output_hidden_states=True).hidden_states
print(hidden_states[-1].shape)
OUT_DIM = hidden_states[-1].shape[-1]
x = F.linear(hidden_states[-1], torch.rand(TARGET_DIM,OUT_DIM))
print(x.shape)
x = F.linear(x.reshape([B,-1,T]), torch.rand(upscale_factor,T))
print(x.shape)
x = x.reshape([B,upscale_factor,-1])
print(x.shape)

x = speecht5_decoder(x)["last_hidden_state"]
print(x.shape)

_, spectogram, logits = post_net(x)
print(spectogram.shape)


text = "Her er en sammenhængende dansk tekst i ét blok: Livet i Danmark er noget helt særligt. Jeg står tidligt op hver morgen og nyder stilheden, før byen vågner. Min lejlighed ligger tæt på havnen, hvor fiskerbådene kommer ind med dagens fangst. Det er blevet en vane at købe frisk fisk direkte fra bådene. Mine naboer er venlige mennesker, som altid hilser, når vi mødes i opgangen. Vi har faktisk en tradition med fællesspisning den første fredag i måneden, hvor alle tager en ret med. Det skaber et godt sammenhold i vores lille fællesskab. Arbejdet fylder meget i hverdagen, men jeg prioriterer også at have tid til familie og venner. Min søster bor i Jylland med sin mand og deres to børn. Vi besøger hinanden så ofte, vi kan. Børnene vokser så hurtigt, og jeg vil nødig gå glip af deres opvækst. I weekenderne cykler jeg ofte langs kysten eller tager en tur i skoven. Naturen giver ro i sindet og nye kræfter til den kommende uge. Selvom det danske vejr kan være omskifteligt, finder man altid skønhed i årstidernes skiften. Efterårets farver er min absolutte favorit."

gemma_tokens = tokenizer.encode(text)
speecht5_tokens = processor(text=text)["input_ids"]

print("Length of tokens")
print("Gemma: ", len(gemma_tokens))
print("SpeechT5: ", len(speecht5_tokens))



# Create a trainable projection layer
class LLMToSpeechT5Adapter(nn.Module):
    def __init__(self, llm_dim=2304, target_dim=768, upscale_factor=4):
        super().__init__()
        self.projection = nn.Linear(llm_dim, target_dim)
        self.upscale_factor = upscale_factor
        self.upscale = nn.Linear(1, upscale_factor)
        # Optional but helpful: add layernorm to normalize representations
        self.norm = nn.LayerNorm(target_dim)
        
    def forward(self, hidden_states):
        B, T, D = hidden_states.shape
        # Project to target dimension
        x = self.projection(hidden_states)  # [B, T, target_dim]
        # Reshape for temporal upscaling
        x = x.transpose(1, 2)  # [B, target_dim, T]
        # Upscale the sequence length
        x = x.unsqueeze(3)  # [B, target_dim, T, 1]
        x = self.upscale(x)  # [B, target_dim, T, upscale_factor]
        x = x.flatten(2)  # [B, target_dim, T*upscale_factor]
        x = x.transpose(1, 2)  # [B, T*upscale_factor, target_dim]
        # Normalize
        x = self.norm(x)
        return x

class LLMToSpeechT5Adapter(nn.Module):
    def __init__(self, llm_dim=2304, target_dim=768, upscale_factor=8):
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
        outputs_before, outputs_after, logits = self.postnet(decoder_outputs.last_hidden_state)
        
        return outputs_after
    
adapter = LLMToSpeechT5Adapter()
llm_to_speech = LLMToSpeech(llm, adapter, speecht5_decoder, post_net)
llm_to_speech.eval()

# Test the model
print(tokens.shape)
output = llm_to_speech(tokens)
print(output.shape)
print(output)

def collate_fn(batch, tokenizer, processor, device="cuda", dtype=torch.bfloat16):
    """
    Custom collate function that processes text inputs for the LLM and audio targets
    for creating mel spectrograms.
    
    Args:
        batch: List of samples from dataset
        tokenizer: LLM tokenizer (Gemma)
        processor: SpeechT5 processor
        device: Device to place tensors on
        dtype: Data type for spectrograms
    """
    # Extract text and audio from batch
    texts = []
    audios = []
    sampling_rates = []
    
    for item in batch:
        # Handle different text formats (string or list)
        text = item["text"]
        if isinstance(text, list):
            text = text[0]
        texts.append(text.lower())
        
        # Get audio and sampling rate
        audios.append(item["audio"]["array"])
        sampling_rates.append(item["audio"]["sampling_rate"])
    
    # Process text inputs in a batch (more efficient)
    tokenized = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Process audio to get mel spectrograms
    mel_specs = []
    mel_lengths = []
    
    # Process audio in parallel if possible
    for audio, sr in zip(audios, sampling_rates):
        # Ensure audio is 1D
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy().flatten()
        else:
            audio = audio.flatten()
            
        # Process using SpeechT5 processor
        processed = processor(audio_target=audio, sampling_rate=sr)
        mel_spec = torch.tensor(processed["input_values"], device=device, dtype=dtype)
        
        # Handle output shape (should be [batch, time, features])
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
            
        mel_specs.append(mel_spec)
        mel_lengths.append(mel_spec.size(1))  # Time dimension
    
    # Pad spectrograms to the same length in the batch
    max_length = max(mel_lengths)
    padded_specs = []
    attention_masks = []
    
    for spec, length in zip(mel_specs, mel_lengths):
        # Create padding if needed
        if length < max_length:
            padding = torch.zeros(
                (spec.size(0), max_length - length, spec.size(2)), 
                device=device, 
                dtype=dtype
            )
            padded = torch.cat([spec, padding], dim=1)
        else:
            padded = spec
            
        # Create attention mask (1 for real values, 0 for padding)
        mask = torch.zeros((spec.size(0), max_length), device=device)
        mask[:, :length] = 1
        
        padded_specs.append(padded)
        attention_masks.append(mask)
    
    # Stack on batch dimension
    mel_spectrograms = torch.cat(padded_specs, dim=0)
    mel_attention_masks = torch.cat(attention_masks, dim=0)
    
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "target_spectrograms": mel_spectrograms,
        "target_attention_mask": mel_attention_masks,
        "target_lengths": torch.tensor(mel_lengths, device=device)
    }


def get_dataloader(dataset, tokenizer, processor, batch_size=8, shuffle=True):
    # Create a CUDA generator for reproducible shuffling
    generator = torch.Generator(device='cuda')
    generator.manual_seed(42)  # Set seed for reproducibility
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,  # Add the CUDA generator
        collate_fn=lambda batch: collate_fn(batch, tokenizer, processor),
        pin_memory=False,  # Already moving to device in collate_fn
    )

# Create dataloader with the custom collate function
train_dataloader = get_dataloader(
    librispeech_dataset_dict["train.clean.100"], 
    tokenizer,
    processor,
    batch_size=8
)

class SpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        
    def forward(self, predicted_specs, target_specs, target_mask=None):
        # Handle different sequence lengths between model output and target
        min_len = min(predicted_specs.size(1), target_specs.size(1))
        pred = predicted_specs[:, :min_len, :]
        target = target_specs[:, :min_len, :]
        
        # Apply mask if provided
        if target_mask is not None:
            mask = target_mask[:, :min_len].unsqueeze(-1)
            loss = self.l1_loss(pred, target) * mask
            # Average over non-masked elements
            return loss.sum() / (mask.sum() + 1e-7)
        else:
            return self.l1_loss(pred, target).mean()

ds = librispeech_dataset_dict["train.clean.360"]

generator = torch.Generator(device='cuda')
loss_fn = SpectrogramLoss()

# Training loop
optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)  # Only train adapter parameters

for epoch in range(1):
    for batch in train_dataloader:
        # Extract batch data
        input_ids = batch["input_ids"]
        target_specs = batch["target_spectrograms"]
        target_mask = batch["target_attention_mask"]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = llm_to_speech(input_ids)
        
        # Calculate loss
        loss = loss_fn(outputs, target_specs, target_mask)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")