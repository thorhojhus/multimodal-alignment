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
    
# Freeze the LLM
for param in llm.parameters():
    param.requires_grad = False
    
# # Optionally freeze parts of SpeechT5's decoder
# # Freezing lower layers and fine-tuning upper layers often works well
# for i, layer in enumerate(speecht5_decoder.layers):
#     if i < 4:  # Freeze first 4 layers for example
#         for param in layer.parameters():
#             param.requires_grad = False

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
        with torch.no_grad():  # Don't need gradients for frozen LLM
            llm_outputs = self.llm(input_ids, output_hidden_states=True)
            hidden_states = llm_outputs.hidden_states[-1]
        
        # Adapt LLM outputs for speech decoder
        adapted_states = self.adapter(hidden_states)
        
        # Pass through decoder
        decoder_outputs = self.decoder(adapted_states)
        
        # Generate spectrograms
        outputs_before, outputs_after, logits = self.postnet(decoder_outputs.last_hidden_state)
        
        return outputs_after
    
adapter = LLMToSpeechT5Adapter()
llm_to_speech = LLMToSpeech(llm, adapter, speecht5_decoder, post_net)
llm_to_speech.eval()

# Test the model
output = llm_to_speech(tokens)
print(output.shape)
print(output)