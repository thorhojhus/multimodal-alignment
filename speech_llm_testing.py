from torch.nn import functional as F
import torch.nn as nn
import transformers
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import torch
from transformers import SpeechT5Model, BitsAndBytesConfig, SpeechT5Processor, SpeechT5HifiGan, AutoTokenizer, SpeechT5ForTextToSpeech

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


from dataset.load_librispeech import librispeech_dataset_dict
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram


import torch.nn as nn
import torch.nn.functional as F

class SpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss(reduction='none')
        
    def forward(self, outputs_after_postnet, target_spectrograms, target_lengths=None, output_lengths=None):
        """
        Args:
            outputs_after_postnet: Model output spectrograms [batch_size, time_steps, n_mels]
            target_spectrograms: Target spectrograms [batch_size, time_steps, n_mels]
            target_lengths: Optional lengths of target sequences
            output_lengths: Optional lengths of output sequences
        """
        # If we don't have explicit lengths, create masks based on non-zero values
        # (assuming padding is zeros or we can create a mask differently)
        if target_lengths is None:
            # Create padding mask (assuming non-zero values are valid)
            padding_mask = target_spectrograms.abs().sum(-1) > 0
        else:
            # Create padding mask from lengths
            max_len = target_spectrograms.size(1)
            batch_size = target_spectrograms.size(0)
            padding_mask = torch.arange(max_len, device=target_spectrograms.device)[None, :] < target_lengths[:, None]
        
        # Handle length differences - use the minimum length between output and target
        min_len = min(outputs_after_postnet.size(1), target_spectrograms.size(1))
        outputs = outputs_after_postnet[:, :min_len]
        targets = target_spectrograms[:, :min_len]
        mask = padding_mask[:, :min_len] if padding_mask.size(1) > min_len else padding_mask
        
        # Calculate L1 loss with masking
        l1_loss = self.l1_criterion(outputs, targets)
        # Apply mask and average
        masked_l1_loss = (l1_loss * mask.unsqueeze(-1)).sum() / (mask.sum() * outputs.size(-1) + 1e-7)
        
        return masked_l1_loss

ds = librispeech_dataset_dict["train.clean.360"]

generator = torch.Generator(device='cuda')
dl = DataLoader(ds, batch_size=1, shuffle=True, generator=generator)
mel = MelSpectrogram(sample_rate=16000, n_mels=80, f_min=80, f_max=7600).to(torch.float32)
# for batch in dl:
#     text = batch["text"]
#     tokens = tokenizer.encode(text[0].lower(), return_tensors="pt")
#     audio = batch["audio"]["array"].to(torch.float32)
#     mel_spec = mel(audio)
#     print("Mel spectrogram shape")
#     print(mel_spec.shape)
#     mel_proc = processor(audio_target=audio.cpu().flatten(), sampling_rate=16000)["input_values"]
#     print("Mel spectrogram shape")
#     print(mel_proc.shape)
#     outs = llm_to_speech(tokens)
#     print("Model output shape")
#     print(outs.shape)

#     break

loss_fn = SpectrogramLoss()

# Training loop
optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)  # Only train adapter parameters

for batch in dl:
    text = batch["text"]
    tokens = tokenizer.encode(text[0].lower(), return_tensors="pt").to("cuda")
    audio = batch["audio"]["array"].to(torch.float32).cuda()
    
    # Get target spectrograms using SpeechT5 processor
    # Note: You might need to adjust this to match the exact format your model outputs
    target_specs = processor(audio_target=audio.cpu().flatten(), sampling_rate=16000)["input_values"]
    target_specs = torch.tensor(target_specs, device="cuda", dtype=torch.bfloat16)
    
    # Forward pass
    optimizer.zero_grad()
    outputs = llm_to_speech(tokens)
    
    # Calculate loss
    loss = loss_fn(outputs, target_specs)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")