# modified from https://github.com/JackismyShephard/hugging-face-audio-course/blob/main/notebooks/training/finetuned-nst-da-training.ipynb
import os
from collections import defaultdict
import torch
from speechbrain.inference import EncoderClassifier
from transformers import SpeechT5Processor
from speechbrain.inference import SpectralMaskEnhancement
from datasets import Audio, load_dataset


min_speaker_counts = 280
max_speaker_counts = 327

device = "cuda" if torch.cuda.is_available() else "cpu"
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
metricgan_model_name = "speechbrain/metricgan-plus-voicebank"
checkpoint = "microsoft/speecht5_tts"

dataset = load_dataset("alexandrainst/nst-da")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

processor = SpeechT5Processor.from_pretrained(checkpoint)
tokenizer = processor.tokenizer
processor_sr = processor.feature_extractor.sampling_rate
assert processor_sr == 16000


dataset = dataset.filter(
    lambda x: not (set("0123456789") & set(x)), input_columns=["text"]
)

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab]}


vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset["train"].column_names,
)

dataset_vocab = set(vocabs["train"]["vocab"][0] + vocabs["test"]["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}

replacements = [
    ("&", "og"),
    ("\r", " "),
    ("´", ""),
    ("\\", ""),
    ("¨", " "),
    ("Å", "AA"),
    ("Æ", "AE"),
    ("É", "E"),
    ("Ö", "OE"),
    ("Ø", "OE"),
    ("á", "a"),
    ("ä", "ae"),
    ("å", "aa"),
    ("è", "e"),
    ("î", "i"),
    ("ô", "oe"),
    ("ö", "oe"),
    ("ø", "oe"),
    ("ü", "y"),
]



def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["text"] = inputs["text"].replace(src, dst)
    return inputs

dataset = dataset.map(cleanup_text, num_proc=24)

speaker_counts = defaultdict(int)

for speaker_id in dataset["train"]["speaker_id"]:
    speaker_counts[speaker_id] += 1

def select_speaker(speaker_id):
    return min_speaker_counts <= speaker_counts[speaker_id] <= max_speaker_counts

dataset["train"] = dataset["train"].filter(
    select_speaker,
    input_columns=["speaker_id"],
)

speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

enhance_model = SpectralMaskEnhancement.from_hparams(
    source=metricgan_model_name,
    savedir=os.path.join("/tmp", metricgan_model_name),
    run_opts={"device": device},
)


def enhance_audio(waveform):
    tensor = torch.tensor(waveform).reshape(1, -1).float()
    enhanced = enhance_model.enhance_batch(tensor, lengths=torch.tensor([1.0]))
    enhanced = enhanced.squeeze().cpu().numpy()
    return enhanced

def prepare_dataset(example, enhance=False):
    audio = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    if enhance:
        audio = enhance_audio(audio)

    example = processor(
        text=example["text"],
        audio_target=audio,
        sampling_rate=sampling_rate,
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio)

    return example

dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 600


dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])

dataset["train"].save_to_disk("/home/thorh/datasets/nst-da/train")
dataset["test"].save_to_disk("/home/thorh/datasets/nst-da/test")