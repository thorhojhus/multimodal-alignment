# %%
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2BertModel,
    AutoProcessor,
    AutoFeatureExtractor,
)
import torch

# Load Wav2Vec2 model and processor
wav2vec_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h", torch_dtype=torch.bfloat16, device_map="cuda"
)
wav2vec_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

# Load Wav2Vec2-BERT model and feature extractor
wav2vec_bert_model = Wav2Vec2BertModel.from_pretrained(
    "facebook/w2v-bert-2.0", torch_dtype=torch.bfloat16, device_map="cuda"
)
wav2vec_bert_processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

# Optional: Set models to evaluation mode
wav2vec_model.eval()
wav2vec_bert_model.eval()

# %%
from datasets import load_dataset

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
)
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate
audio_array = dataset[0]["audio"]["array"]


import numpy as np


def pad_audio_to_30_seconds(audio_array, sampling_rate):
    """
    Pad or trim audio array to exactly 30 seconds.

    Args:
        audio_array (np.ndarray): Input audio array
        sampling_rate (int): Sampling rate of the audio

    Returns:
        np.ndarray: Padded/trimmed audio array of exactly 30 seconds
    """
    target_length = 30 * sampling_rate  # 30 seconds * sampling_rate
    current_length = len(audio_array)

    if current_length < target_length:
        # Pad with zeros if audio is shorter than 30 seconds
        padding_length = target_length - current_length
        padded_array = np.pad(audio_array, (0, padding_length), mode="constant")
        return padded_array
    else:
        # Trim if audio is longer than 30 seconds
        return audio_array[:target_length]


# Apply the padding to your audio
padded_audio = pad_audio_to_30_seconds(audio_array, sampling_rate)

# Verify the length
print(f"Original duration: {len(audio_array) / sampling_rate:.2f} seconds")
print(f"Padded duration: {len(padded_audio) / sampling_rate:.2f} seconds")

# %%
wav2vec_inputs = wav2vec_processor(
    padded_audio, sampling_rate=sampling_rate, return_tensors="pt"
)
wav2vec_inputs = {k: v.to("cuda", torch.bfloat16) for k, v in wav2vec_inputs.items()}

# %%
with torch.no_grad():
    wav2vec_outputs = wav2vec_model(**wav2vec_inputs)

# print
print(
    "Shape of the last hidden state for wav2vec_model:",
    wav2vec_outputs.last_hidden_state.shape,
)
print(
    "Shape of extract_features output for wav2vec_model:",
    wav2vec_outputs.extract_features.shape,
)

# %%
wav2vec_bert_inputs = wav2vec_bert_processor(
    padded_audio, sampling_rate=sampling_rate, return_tensors="pt"
)
wav2vec_bert_inputs = {
    "input_features": wav2vec_bert_inputs["input_features"].cuda().to(torch.bfloat16),
    "attention_mask": wav2vec_bert_inputs["attention_mask"].cuda().to(torch.int32),
}


with torch.no_grad():
    wav2vec_bert_outputs = wav2vec_bert_model(**wav2vec_bert_inputs)

print(
    "Shape of the last hidden state for wav2vec_bert_model:",
    wav2vec_bert_outputs.last_hidden_state.shape,
)
print(
    "Shape of extract_features output for wav2vec_bert_model:",
    wav2vec_bert_outputs.extract_features.shape,
)

# %%
from transformers import WhisperModel, WhisperProcessor

# Load Whisper model and processor
whisper_model = WhisperModel.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch.bfloat16, device_map="cuda"
)
whisper_model.eval()
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# Process audio input
whisper_inputs = whisper_processor(
    padded_audio, sampling_rate=sampling_rate, return_tensors="pt"
)
whisper_inputs = {k: v.to("cuda", torch.bfloat16) for k, v in whisper_inputs.items()}

# Get encoder outputs
with torch.no_grad():
    whisper_outputs = whisper_model.get_encoder()(**whisper_inputs)

# Print shapes of encoder outputs
print(
    "Shape of the encoder last hidden state for whisper-large-v3:",
    whisper_outputs.last_hidden_state.shape,
)


