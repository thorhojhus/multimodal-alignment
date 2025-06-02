import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
import re
import functools
from tqdm import tqdm
from collections import defaultdict
import json
from models.audio_llm import AudioLLMModel
from types import SimpleNamespace
import torch
from transformers import AutoFeatureExtractor, AutoTokenizer


CHECKPOINT_PATH = "/work3/s224233/shared/outputs/2025-05-14/02-05-05/checkpoints/checkpoint_latest.pt"


torch.set_float32_matmul_precision('high')

# Config
config = SimpleNamespace(
    model=SimpleNamespace(
        lm_model_name="google/gemma-3-4b-it",
        encoder_model_name="facebook/w2v-bert-2.0",
        downsample_factor=8,
        load_in_4bit=False,
        audio_tokenizer=False,
        projector="conv",
        train_encoder_full=True
    ),
    training=SimpleNamespace(kl_loss=False),
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

model = AudioLLMModel(config)
checkpoint = torch.load(
    CHECKPOINT_PATH
)

print(checkpoint.keys())
model.projector.load_state_dict(checkpoint["CNNProjector"])
model.audio_encoder.audio_encoder.load_state_dict(checkpoint["Wav2Vec2BertModel"])
model.eval()

def create_conversation(category, audio_token="<|audio|>"):
    """Create conversation messages for a given category."""

    system_prompt = """You are a thorough logical reasoner. Your responses directly determine evaluation accuracy.

Critical rules for success:
- Listen to the ENTIRE problem before starting your analysis
- State the problem completely
- Work through each step methodically - rushing leads to errors
- Double-check your work before finalizing
- Count carefully and track all details
- Trust your systematic reasoning over first impressions

Common pitfalls to avoid:
- Missing negations ("not", "except", "unless")
- Confusing similar-sounding words or numbers
- Losing track of your position/count/logic state
- Making assumptions beyond what's explicitly stated
- Jumping to a conclusion and not checking your answers

Remember: You excel at these tasks when you're careful and systematic. Take your time don't just give an answer."""

    category_instructions = {
        "formal_fallacies": f"""Listen to the logical argument: \n\n{audio_token}
Think step by step:
1. Identify the premises and conclusion
2. Check if the conclusion logically follows from the premises  
3. Determine if the argument structure is valid

Provide your reasoning, then give your final answer as <answer>valid</answer> or <answer>invalid</answer>
""",

        "navigate": f"""Listen to the navigation instructions: \n\n{audio_token}

Think step by step:
1. Start at the origin point
2. Follow each movement instruction carefully
3. Track your final position
4. Determine if you end up back at the starting point

Provide your reasoning, then give your final answer as <answer>yes</answer> or <answer>no</answer>""",

        "object_counting": f"""Listen to the description: \n\n{audio_token}

Think step by step:
1. List all objects mentioned
2. Identify which objects belong to the category being asked about
3. Count the total number

Provide your reasoning, then give your final answer as <answer>NUMBER</answer>""",

        "web_of_lies": f"""Listen to the logic puzzle: \n\n{audio_token}

Think step by step:
1. Identify who tells truth and who lies based on the rules
2. Work through the logical implications
3. Determine if the final statement can be answered with yes or no.

Provide your reasoning, then give your final answer as <answer>yes</answer> or <answer>no</answer>"""
    }
    
    instruction = category_instructions.get(
        category.lower(),
        f"""Listen to the audio: {audio_token}

Think step by step and explain your reasoning, then provide your final answer in <answer> tags."""
    )

    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]

class BigBenchAudioDataset(Dataset):
    def __init__(self, tokenizer, processor, split="train", audio_token="<|audio|>"):
        """Initialize the Big Bench Audio dataset wrapper.
        
        Args:
            tokenizer: Tokenizer for the language model
            processor: Audio processor/feature extractor
            split: Dataset split to use (default: train)
            audio_token: Special token for audio placeholder
        """
        self.dataset = load_dataset("ArtificialAnalysis/big_bench_audio")[split]
        self.tokenizer = tokenizer
        self.processor = processor
        self.audio_token = audio_token

        # Add audio token if not in vocabulary
        if self.audio_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_token]})

        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)

    def __len__(self):
        return len(self.dataset)

    def _audio_to_features(self, audio_array, sampling_rate):
        """Process audio array into features."""
        # Resample if necessary (w2v-bert expects 16kHz)
        target_sampling_rate = 16000
        if sampling_rate != target_sampling_rate:
            import torchaudio
            audio_array = torchaudio.functional.resample(
                torch.tensor(audio_array),
                orig_freq=sampling_rate,
                new_freq=target_sampling_rate
            ).numpy()

        # Process with w2v-bert feature extractor
        return self.processor(
            audio_array,
            sampling_rate=target_sampling_rate,
            return_tensors="pt",
            padding=True,
        )

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Extract fields
        category = example["category"]
        official_answer = example["official_answer"].lower()
        audio = example["audio"]
        sample_id = example["id"]

        # Handle audio format
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
        else:
            # Default to 16kHz if not specified
            audio_array = np.array(audio)
            sampling_rate = 16000

        # Convert to float32 tensor
        audio_array = torch.tensor(audio_array, dtype=torch.float32)

        # Process audio features
        audio_features = self._audio_to_features(audio_array.numpy(), sampling_rate)

        # Format chat with audio token for generation (user message only)
        conversation_gen = create_conversation(category, self.audio_token)

        # Tokenize for generation (without answer)
        tokenized_gen = self.tokenizer.apply_chat_template(
            conversation_gen,
            add_generation_prompt=True,  # Add generation prompt for inference
            return_tensors="pt"
        )[0]

        return {
            "input_ids": tokenized_gen,
            "audio_features": audio_features,
            "category": category,
            "official_answer": official_answer,
            "id": sample_id,
            "raw_audio": audio_array,  # Keep raw audio for debugging
        }

# Custom collate function that preserves metadata
def collate_fn_with_metadata(batch, tokenizer, T=30):
    # Separate metadata from data needed for standard collate_fn
    categories = [item["category"] for item in batch]
    official_answers = [item["official_answer"] for item in batch]
    ids = [item["id"] for item in batch]
    raw_audios = [item["raw_audio"] for item in batch]

    # Create dummy fields needed by standard collate_fn
    for item in batch:
        item["labels"] = torch.full_like(item["input_ids"], -100)  # Dummy labels
        item["alt_input_ids"] = item["input_ids"].clone()  # Dummy alt
        item["alt_labels"] = item["labels"].clone()  # Dummy alt labels
        item["text_ids"] = torch.tensor([0])  # Dummy text ids

    # Use standard collate_fn
    from dataset.audio_dataset import collate_fn
    collated = collate_fn(batch, tokenizer, T)

    # Add metadata
    collated["category"] = categories
    collated["official_answer"] = official_answers
    collated["id"] = ids
    collated["raw_audio"] = raw_audios

    return collated

def extract_answer(text, category=None):
    """Improved answer extraction with category-specific fallbacks."""
    
    # First try to find explicit answer tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip().lower()
    
    # Category-specific extraction as fallback
    text_lower = text.lower()
    
    if category == "formal_fallacies":
        if "invalid" in text_lower:
            return "invalid"
        elif "valid" in text_lower:
            return "valid"
    
    elif category == "navigate":
        if "no" in text_lower:
            return "no"
        elif "yes" in text_lower:
            return "yes"
    
    elif category == "web_of_lies":
        if "false" in text_lower:
            return "false"
        elif "true" in text_lower:
            return "true"
    
    elif category == "object_counting":
        # Extract the last number mentioned
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return numbers[-1]
    
    # Final fallback: look for direct answers in the last sentence
    sentences = text.split('.')
    if sentences:
        final_sentence = sentences[-1].strip().lower()
        for pattern in ["valid", "invalid", "yes", "no", "true", "false"]:
            if pattern in final_sentence:
                return pattern
    
    return "unknown"

def evaluate_big_bench_audio(model, dataloader, tokenizer, max_new_tokens=1024):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                # Reconstruct conversations for the batch using the standalone function
                conversations = [
                    create_conversation(category, "<|audio|>") 
                    for category in batch["category"]
                ]
                
                # Use apply_chat_template for the entire batch
                tokenized_batch = tokenizer.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    padding=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                user_prompts = tokenized_batch["input_ids"].to("cuda")
                user_prompt_masks = tokenized_batch["attention_mask"].to("cuda")
                
                # Handle audio features
                audio_features = batch["audio_features"]
                if isinstance(audio_features, dict):
                    if "input_features" in audio_features:
                        audio_features = {
                            'input_features': audio_features["input_features"].to("cuda", torch.bfloat16),
                            'attention_mask': audio_features["attention_mask"].to("cuda")
                        }
                    elif "input_values" in audio_features:
                        audio_features = {
                            'input_values': audio_features["input_values"].to("cuda", torch.bfloat16)
                        }
                    else:
                        audio_features = {k: v.to("cuda", torch.bfloat16) for k, v in audio_features.items()}
                else:
                    audio_features = audio_features.to("cuda", torch.bfloat16)

                insertion_indices = [idx.to("cuda") for idx in batch["insertion_indices"]]

                # Generate using your existing approach
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=user_prompts,
                        audio_features=audio_features,
                        insertion_indices=insertion_indices,
                        attention_mask=user_prompt_masks,
                        max_new_tokens=max_new_tokens,
                    )

                # Decode only the generated part
                for i, output in enumerate(outputs):
                    # Get only the newly generated tokens
                    model_response = tokenizer.decode(output, skip_special_tokens=True)
                    
                    model_response = model_response.strip()

                    # Extract answer with category context
                    category = batch["category"][i]
                    extracted_answer = extract_answer(model_response, category)

                    results.append({
                        "id": batch["id"][i],
                        "category": category,
                        "prediction": extracted_answer,
                        "official_answer": batch["official_answer"][i],
                        "full_response": model_response,
                    })

                    print(f"\nID: {batch['id'][i]}")
                    print(f"Category: {category}")
                    print(f"Model Response: {model_response}")
                    print(f"Extracted Answer: {extracted_answer}")
                    print(f"Ground Truth: {batch['official_answer'][i]}")
                    print(f"Correct: {extracted_answer == batch['official_answer'][i]}")
                    print("-" * 80)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error, skipping batch. Error: {e}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    return results

# Main evaluation execution
if __name__ == "__main__":
    # Create the dataset (this should only happen once)
    print("Loading dataset...")
    big_bench_dataset = BigBenchAudioDataset(
        tokenizer=tokenizer,
        processor=processor,
        split="train"  # Big Bench Audio only has train split
    )

    # Create dataloader with reduced batch size
    batch_size = 8
    dataloader = DataLoader(
        big_bench_dataset,
        batch_size=batch_size,
        collate_fn=functools.partial(collate_fn_with_metadata, tokenizer=tokenizer, T=30),
        shuffle=False
    )

    # Clear cache before evaluation
    torch.cuda.empty_cache()

    # Run evaluation
    print("Starting Big Bench Audio evaluation...")

    results = evaluate_big_bench_audio(model, dataloader, tokenizer, max_new_tokens=1024)

    # Calculate accuracy by category
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    for result in results:
        category = result["category"]
        is_correct = result["prediction"] == result["official_answer"]

        category_correct[category] += int(is_correct)
        category_total[category] += 1
        category_correct["overall"] += int(is_correct)
        category_total["overall"] += 1

    # Print results
    print(f"\nEvaluated {len(results)} samples")
    print("\nAccuracy by category:")
    for category in sorted(category_total.keys()):
        if category == "overall":
            continue
        accuracy = category_correct[category] / category_total[category] * 100
        print(f"{category}: {accuracy:.1f}% ({category_correct[category]}/{category_total[category]})")

    print(f"\nOverall accuracy: {category_correct['overall'] / category_total['overall'] * 100:.1f}% ({category_correct['overall']}/{category_total['overall']})")

    # Save results
    with open("big_bench_audio_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print some detailed examples
    print("\nDetailed examples:")
    for i in range(min(3, len(results))):
        print(f"\n{'='*80}")
        print(f"Example {i+1}:")
        print(f"Category: {results[i]['category']}")
        print(f"Full Response:\n{results[i]['full_response']}")
        print(f"\nExtracted Answer: {results[i]['prediction']}")
        print(f"Ground Truth: {results[i]['official_answer']}")
        print(f"Correct: {results[i]['prediction'] == results[i]['official_answer']}")

    print("\nEvaluation complete!")