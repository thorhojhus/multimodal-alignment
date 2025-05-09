import json
import os
import glob
from typing import Dict, List, Any, Tuple
from datasets import DatasetDict, Dataset, concatenate_datasets
from tqdm import tqdm

# vLLM imports
from vllm import LLM, SamplingParams

import truecase

def setup_vllm_model():
    """Set up the vLLM model with Gemma 3 4B-it"""
    print("Loading Gemma 3 model with vLLM...")
    
    model = LLM(
        model="google/gemma-3-4b-pt",
        enable_chunked_prefill=True,
        max_model_len=256,
        max_seq_len_to_capture=256,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
    )
    
    return model

def process_dataset_batch(
    model: LLM,
    texts: List[str], 
    ids: List[str],
    sampling_params: SamplingParams
) -> Dict[str, Dict[str, str]]:
    """Process a batch of texts using vLLM's efficient batching"""
    
    # Generate continuations with vLLM's optimized batch processing
    outputs = model.generate(texts, sampling_params)
    
    # Extract continuations and match them with IDs
    results = {}
    for output, id, truecased_text in zip(outputs, ids, texts):
        # Get only the generated text (not the original input)
        continuation = output.outputs[0].text
        results[id] = {
            "text_truecased": truecased_text,
            "text_continuation": continuation
        }
    
    return results

def generate_and_save_continuations(
    dataset_dict: DatasetDict,
    output_dir: str,
    batch_size: int = 256,
    splits_to_process: List[str] = ["test", "train"]  # Only process test and train splits
):
    """Generate continuations for LibriSpeech dataset splits and save to JSON"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the vLLM model
    model = setup_vllm_model()
    
    # Set optimal sampling parameters for Gemma 3
    sampling_params = SamplingParams(
        max_tokens=64,
    )
    
    # Process each split
    for split_name, dataset in dataset_dict.items():
        # Skip splits that are not test or train
        if split_name not in splits_to_process:
            print(f"Skipping {split_name} split (not in {splits_to_process})")
            continue
            
        print(f"Processing {split_name} with {len(dataset)} examples...")
        
        # Check if file already exists to support resuming
        output_file = os.path.join(output_dir, f"{split_name}_continuations.json")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
            print(f"Found existing continuations for {split_name}, with {len(results)} entries")
        else:
            results = {}
        
        # Get IDs that still need processing
        remaining_ids = [id for id in dataset["id"] if id not in results]
        if not remaining_ids:
            print(f"All examples in {split_name} already processed. Skipping.")
            continue
        
        # Filter dataset to only process remaining examples
        indices = [i for i, id in enumerate(dataset["id"]) if id in remaining_ids]
        remaining_dataset = dataset.select(indices)
        
        # Process in batches for efficiency
        for i in tqdm(range(0, len(remaining_dataset), batch_size), desc=f"Processing {split_name}"):
            # Get batch
            batch_indices = range(i, min(i + batch_size, len(remaining_dataset)))
            batch = remaining_dataset.select(batch_indices)
            
            # Use text field as specified
            batch_texts_raw = batch["text"]
            batch_texts = []
            
            # Apply truecasing to the texts
            for text in batch_texts_raw:
                truecased_text = truecase.get_true_case(text)
                batch_texts.append(truecased_text)
            
            batch_ids = batch["id"]
            
            # Process batch with vLLM
            batch_results = process_dataset_batch(
                model, batch_texts, batch_ids, sampling_params
            )
            
            # Update results dictionary
            results.update(batch_results)
            
            # Save incrementally to handle interruptions
            with open(output_file, 'w') as f:
                json.dump(results, f)
        
        print(f"Finished processing {split_name}. Saved {len(results)} entries to {output_file}")

def add_continuations_to_dataset(dataset, continuations_file):
    """Add continuations and truecased text as columns to a dataset by matching IDs"""
    
    # Load continuations
    with open(continuations_file, 'r') as f:
        results_dict = json.load(f)
    
    # Define mapping function
    def add_continuation_and_truecased(example):
        example_id = example["id"]
        if example_id in results_dict:
            example["text_truecased"] = results_dict[example_id].get("text_truecased", "")
            example["text_continuation"] = results_dict[example_id].get("text_continuation", "")
        else:
            example["text_truecased"] = ""
            example["text_continuation"] = ""
        return example
    
    # Add the columns
    return dataset.map(add_continuation_and_truecased)

def main():
    from dataset.load_librispeech import librispeech_dataset_dict
    
    # Generate and save continuations (only for test and train splits)
    continuations_dir = "./dataset/librispeech_continuations"
    generate_and_save_continuations(
        librispeech_dataset_dict, 
        continuations_dir,
        splits_to_process=["test", "train"]  # Explicitly specify splits
    )
    
    print("Finished generating continuations for test and train splits!")

if __name__ == "__main__":
    main()