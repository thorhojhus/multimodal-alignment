import transformers
import torch
from jiwer import wer
from tqdm import tqdm  # Progress bar
from dataset.load_coral import coral_dataset_dict

# Load the model and processor
model = transformers.SeamlessM4Tv2ForSpeechToText.from_pretrained(
    "facebook/seamless-m4t-v2-large", 
    device_map="cuda", 
    torch_dtype=torch.bfloat16
)
model = torch.compile(model)

processor = transformers.AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

# Use the test set from LibriSpeech
dataset = coral_dataset_dict["test"]

# Set batch size
batch_size = 64

total_wer = 0.0
num_samples = 0

# Loop over the dataset in batches with tqdm progress bar
for i in range(0, len(dataset), batch_size):
    batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

    # Prepare lists of audio arrays and their corresponding reference texts
    audio_arrays = [sample["audio"]["array"] for sample in batch]
    references = [sample["text"].lower() for sample in batch]

    # Process the batch of audio arrays into model input features
    inputs = processor(audios=audio_arrays, return_tensors="pt", sampling_rate=16000)
    inputs = {k: v.to("cuda", torch.bfloat16, non_blocking=True) for k, v in inputs.items()}
    
    # Run inference
    with torch.inference_mode():
        tokens = model.generate(inputs["input_features"])
    
    # Decode the model outputs
    predictions = processor.batch_decode(tokens, skip_special_tokens=True)
    for pred, ref in zip(predictions, references):
        print("ref:")
        print(ref)
        print("pred:")
        print(pred)
    sample_wer = wer(references, predictions)
    print(sample_wer)
    total_wer += sample_wer
    num_samples += 1
    #torch.cuda.empty_cache()

# Calculate and print the average WER
average_wer = total_wer / num_samples
print("\nAverage WER:", average_wer)
