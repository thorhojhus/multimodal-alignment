import torch
import torch.nn as nn
from types import SimpleNamespace
from models.audio_llm import AudioLLMModel

def test_audio_llm_with_mimi():
    """
    Test the full AudioLLMModel with Mimi encoder integration.
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    print("1. Setting up configuration...")
    # Set up a simple configuration
    config = SimpleNamespace(
        model=SimpleNamespace(
            encoder_model_name="kyutai/mimi",
            lm_model_name="HuggingFaceTB/gemma-2-2b-it",  # A small model for testing
            audio_tokenizer=False,
            downsample_factor=5,
            projector="linear",
            load_in_4bit=True
        ),
        training=SimpleNamespace(
            loss="ce",  # Cross-entropy loss
            kl_temp=1.0
        )
    )
    
    print("2. Initializing AudioLLMModel...")
    try:
        # Initialize the model
        model = AudioLLMModel(config)
        
        print("3. Creating mock inputs...")
        batch_size = 2
        seq_len = 20
        
        # Create input text tokens
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device="cuda", dtype=torch.long)
        
        # Insert audio token at a specific position
        input_ids[0, 5] = model.audio_token_id
        input_ids[1, 3] = model.audio_token_id
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Create mock audio input
        mock_audio = {"input_values": torch.randn(batch_size, 1, 16000, device="cuda", dtype=torch.bfloat16)}
        
        # Get insertion indices (positions of audio tokens)
        insertion_indices = []
        for i in range(batch_size):
            indices = (input_ids[i] == model.audio_token_id).nonzero()
            insertion_indices.append(indices)
        
        # Create labels (same as input_ids for causal LM training)
        labels = input_ids.clone()
        
        print("4. Running forward pass...")
        # Forward pass
        with torch.no_grad():
            output = model.forward(
                input_ids=input_ids,
                audio_features=mock_audio,
                insertion_indices=insertion_indices,
                attention_mask=attention_mask,
                labels=labels
            )
        
        print("5. Testing generation...")
        # Test generation
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                audio_features=mock_audio,
                insertion_indices=insertion_indices,
                attention_mask=attention_mask,
                max_new_tokens=5
            )
        
        print("\nSuccess! Full model with Mimi is working correctly.")
        print(f"Loss: {output.loss.item()}")
        print(f"Generated shape: {generated.shape}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_llm_with_mimi()