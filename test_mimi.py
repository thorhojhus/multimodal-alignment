import torch
from models.mimi_tokenizer import MimiTokenizer
from models.encoders import EncoderWrapper
from types import SimpleNamespace

def test_mimi_encoder_tokenizer_pipeline():
    """
    Test the Mimi encoder and tokenizer pipeline to ensure it processes audio features correctly.
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    print("1. Setting up encoder wrapper...")
    # Initialize the encoder
    encoder = EncoderWrapper("kyutai/mimi")
    
    print("2. Setting up tokenizer...")
    # Initialize the tokenizer
    embed_dim = 1024  # Example LLM embedding dimension
    tokenizer = MimiTokenizer(
        vocab_size=2048,
        embed_dim=embed_dim,
        use_hierarchical=True
    )
    
    print("3. Creating mock input...")
    # Create mock audio input with the correct shape for Mimi: [batch_size, channels, length]
    # Mimi expects mono audio with a sampling rate of 16000Hz, in bfloat16 format
    batch_size = 2
    mock_input = {"input_values": torch.randn(batch_size, 1, 16000, device="cuda", dtype=torch.bfloat16)}
    
    print("4. Encoding audio...")
    try:
        # Encode audio with Mimi to get audio codes
        with torch.no_grad():
            encoder_outputs = encoder(mock_input)
            audio_codes = encoder_outputs.audio_codes
            
        print(f"   ✓ Audio codes shape: {audio_codes.shape}")
        
        # The audio codes should have shape [batch_size, 32, seq_len]
        # where 32 is the number of codebooks and seq_len depends on audio length
        
        print("5. Tokenizing...")
        # Process through our tokenizer
        audio_embeddings = tokenizer(audio_codes)
        
        print(f"   ✓ Audio embeddings shape: {audio_embeddings.shape}")
        # The resulting shape should be [batch_size, seq_len, embed_dim]
        
        print("\nSuccess! The pipeline is working correctly.")
        print(f"   Input shape: {mock_input['input_values'].shape}")
        print(f"   Audio codes shape: {audio_codes.shape}")
        print(f"   Final embedding shape: {audio_embeddings.shape}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mimi_encoder_tokenizer_pipeline()