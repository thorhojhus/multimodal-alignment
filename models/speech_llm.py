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
    
# Optionally freeze parts of SpeechT5's decoder
# Freezing lower layers and fine-tuning upper layers often works well
for i, layer in enumerate(speecht5_decoder.layers):
    if i < 4:  # Freeze first 4 layers for example
        for param in layer.parameters():
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