import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from jiwer import wer, cer, compute_measures
import wandb
from joblib import Parallel, delayed
import os
from datetime import datetime

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# Connect to the specific run
api = wandb.Api()
runs = api.runs("borkeren/benchmarks")

def calculate_single_bootstrap(gts, preds, indices):
    """Calculate WER and CER for a single bootstrap sample"""
    # Sample with replacement
    sample_indices = np.random.choice(indices, size=len(indices), replace=True)
    
    # Get sampled ground truths and predictions
    sampled_gts = [gts[idx] for idx in sample_indices]
    sampled_preds = [preds[idx] for idx in sample_indices]
    
    # Calculate WER and CER on concatenated texts
    return wer(sampled_gts, sampled_preds), cer(sampled_gts, sampled_preds)

def parallel_bootstrap_confidence_interval(gts, preds, n_bootstrap=1000, confidence=0.95, n_jobs=-1):
    """
    Calculate bootstrap confidence intervals for WER and CER using parallel processing
    """
    indices = np.arange(len(gts))
    
    # Run bootstrap iterations in parallel
    print(f"Running {n_bootstrap} bootstrap iterations in parallel...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_single_bootstrap)(gts, preds, indices) 
        for _ in tqdm(range(n_bootstrap))
    )
    
    # Extract results
    wer_bootstrap = np.array([r[0] for r in results])
    cer_bootstrap = np.array([r[1] for r in results])
    
    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    
    wer_mean = np.mean(wer_bootstrap)
    wer_ci_low = np.percentile(wer_bootstrap, 100 * alpha)
    wer_ci_high = np.percentile(wer_bootstrap, 100 * (1 - alpha))
    
    cer_mean = np.mean(cer_bootstrap)
    cer_ci_low = np.percentile(cer_bootstrap, 100 * alpha)
    cer_ci_high = np.percentile(cer_bootstrap, 100 * (1 - alpha))
    
    return (wer_mean, wer_ci_low, wer_ci_high), (cer_mean, cer_ci_low, cer_ci_high)

def normalize_text(text):
    """
    Normalize text by removing punctuation and converting to lowercase.
    """
    return ''.join(e for e in text if e.isalnum() or e.isspace()).lower()

def save_results_to_file(results, filename="speech_recognition_results.txt"):
    """
    Save results to a nicely formatted text file
    
    Parameters:
    - results: List of result dictionaries
    - filename: Name of the output file
    """
    with open(filename, 'w') as f:
        # Write header with timestamp
        f.write(f"Speech Recognition Evaluation Results\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Write each result
        for result in results:
            f.write(f"Run ID: {result['run_id']}\n")
            f.write(f"Model: {result['model_name']}\n")
            f.write("-"*80 + "\n")
            
            # Bootstrap results
            f.write("Bootstrap Results (95% Confidence Intervals):\n")
            f.write(f"WER: {result['wer_mean']:.4f} ({result['wer_ci_low']:.4f} - {result['wer_ci_high']:.4f})\n")
            f.write(f"CER: {result['cer_mean']:.4f} ({result['cer_ci_low']:.4f} - {result['cer_ci_high']:.4f})\n\n")
            
            # Total results
            f.write("Total Results:\n")
            f.write(f"WER: {result['total_wer']:.4f}\n")
            f.write(f"CER: {result['total_cer']:.4f}\n\n")
            
            f.write("="*80 + "\n\n")
            
    print(f"Results saved to {filename}")

def save_results_to_csv(results, filename="speech_recognition_results.csv"):
    """
    Save results to a CSV file
    
    Parameters:
    - results: List of result dictionaries
    - filename: Name of the output file
    """
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    columns = [
        'model_name', 'run_id', 
        'wer_mean', 'wer_ci_low', 'wer_ci_high',
        'cer_mean', 'cer_ci_low', 'cer_ci_high',
        'total_wer', 'total_cer'
    ]
    
    # Save to CSV
    df[columns].to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Main code
results = []

for run in runs:
    # To get artifacts from each run
    artifacts = run.logged_artifacts()

    for artifact in reversed(artifacts):
        if "evaluation_completions" in artifact.name:
            if artifact.size > 400000:
                print(f"Downloading artifact: {artifact.name}")
                artifact_dir = artifact.download()
                print(f"Downloaded to: {artifact_dir}")
                data = load_json(f"{artifact_dir}/evaluation_completions.table.json")
                df = pd.DataFrame(data["data"], columns=data["columns"])
                print(f"Run ID: {run.id}, Name: {run.name}")
                
                # Prepare data
                gts = []
                preds = []
                for i in range(len(df)):
                    gt, pred = normalize_text(df["Ground Truth"][i]), normalize_text(df["Generated"][i])
                    gts.append(gt)
                    preds.append(pred)
                
                # Calculate total WER/CER once (for validation)
                print("Calculating total WER/CER...")
                total_wer = wer(gts, preds)
                total_cer = cer(gts, preds)
                
                # Use the parallel bootstrap function
                print("Running parallel bootstrap...")
                (wer_mean, wer_ci_low, wer_ci_high), (cer_mean, cer_ci_low, cer_ci_high) = parallel_bootstrap_confidence_interval(gts, preds)
                
                # Print results
                print("Bootstrap Results (95% Confidence Intervals):")
                print(f"WER: {wer_mean:.4f} ({wer_ci_low:.4f} - {wer_ci_high:.4f})")
                print(f"CER: {cer_mean:.4f} ({cer_ci_low:.4f} - {cer_ci_high:.4f})")
                print(f"Total WER: {total_wer:.4f}")
                print(f"Total CER: {total_cer:.4f}")
                
                # Store results
                result = {
                    'run_id': run.id,
                    'model_name': run.name,
                    'wer_mean': wer_mean,
                    'wer_ci_low': wer_ci_low,
                    'wer_ci_high': wer_ci_high,
                    'cer_mean': cer_mean,
                    'cer_ci_low': cer_ci_low,
                    'cer_ci_high': cer_ci_high,
                    'total_wer': total_wer,
                    'total_cer': total_cer
                }
                results.append(result)
                
                break

# Save all results after processing
if results:
    save_results_to_file(results)
    save_results_to_csv(results)
    
    # Also generate a LaTeX table
    with open("speech_recognition_latex_table.txt", 'w') as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{WER} & \\textbf{CER} \\\\\n")
        f.write("\\midrule\n")
        
        for result in results:
            model_name = result['model_name']
            wer_str = f"{result['wer_mean']:.4f} ({result['wer_ci_low']:.4f} - {result['wer_ci_high']:.4f})"
            cer_str = f"{result['cer_mean']:.4f} ({result['cer_ci_low']:.4f} - {result['cer_ci_high']:.4f})"
            
            f.write(f"{model_name} & {wer_str} & {cer_str} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    
    print("LaTeX table saved to speech_recognition_latex_table.txt")