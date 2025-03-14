import os
import torchaudio
from models.SparkTTS import SparkTTS

model_wrap = SparkTTS(os.environ["HF_HOME"], True, "cuda")

output_wavs = model_wrap.inference(
    ["Hello, this is a test sentence. I am testing the SparkTTS model", "This is another test sentence", "And right here is a danish sentence. Jeg elsker at spise is!"],
    gender="male",
    speed="moderate",
    pitch="moderate"
)
print("done, saving now ...")

for i, wav in enumerate(output_wavs):
    torchaudio.save(f"tts-wavs/output_{i}.wav", wav.unsqueeze(0), 16000)