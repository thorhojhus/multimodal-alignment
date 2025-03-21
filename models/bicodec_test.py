from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.models.bicodec import BiCodec
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP
from pathlib import Path
import torch

# Set device to CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = BiCodec.load_from_checkpoint("weights/BiCodec")
model = model.to(device)  # Move model to GPU

# Generate random inputs for testing
duration = 0.96*4
x = torch.randn(32, 1, int(duration * 16000), device=device)
feat = torch.randn(32, int(duration * 50), 1024, device=device)
inputs = {"feat": feat, "wav": x, "ref_wav": x}

with torch.inference_mode():
   outputs = model(inputs)
   
# Forward pass
import time
start = time.time()
for i in range(10):
   with torch.inference_mode():
       outputs = model(inputs)
stop = time.time()

print("Time taken for model out: ", stop - start)

with torch.inference_mode():
   model.tokenize(inputs)

start_time = time.time()
for i in range(10):
   with torch.inference_mode():
       semantic_tokens, global_tokens = model.tokenize(inputs)
stop_time = time.time()

print("Time taken for tokenization:", stop_time - start_time)

print(outputs["x_vector"].shape, outputs["d_vector"].shape)
print(semantic_tokens.shape, global_tokens.shape)