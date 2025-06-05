import pandas as pd
import numpy as np
from jiwer import wer, cer
from joblib import Parallel, delayed
from tqdm import tqdm
import os

def calculate_single_bootstrap(refs, preds, indices):
    """Calculate WER and CER for a single bootstrap sample"""
    sample_indices = np.random.choice(indices, size=len(indices), replace=True)
    sampled_refs = [refs[idx] for idx in sample_indices]
    sampled_preds = [preds[idx] for idx in sample_indices]
    return wer(sampled_refs, sampled_preds), cer(sampled_refs, sampled_preds)

def bootstrap_confidence_interval(refs, preds, n_bootstrap=1000, confidence=0.95, n_jobs=None):
    """Calculate bootstrap confidence intervals for WER and CER using parallel processing"""
    if n_jobs is None:
        # Use half the available cores
        n_jobs = max(1, os.cpu_count() // 2)
    
    indices = np.arange(len(refs))
    
    # Run bootstrap iterations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_single_bootstrap)(refs, preds, indices) 
        for _ in tqdm(range(n_bootstrap), desc="Bootstrap iterations")
    )
    
    # Extract WER and CER results
    wer_bootstrap = np.array([r[0] for r in results])
    cer_bootstrap = np.array([r[1] for r in results])
    
    # Calculate mean and confidence interval for WER
    wer_mean = np.mean(wer_bootstrap)
    alpha = (1 - confidence) / 2
    wer_ci_low = np.percentile(wer_bootstrap, 100 * alpha)
    wer_ci_high = np.percentile(wer_bootstrap, 100 * (1 - alpha))
    wer_half_width = (wer_ci_high - wer_ci_low) / 2
    
    # Calculate mean and confidence interval for CER
    cer_mean = np.mean(cer_bootstrap)
    cer_ci_low = np.percentile(cer_bootstrap, 100 * alpha)
    cer_ci_high = np.percentile(cer_bootstrap, 100 * (1 - alpha))
    cer_half_width = (cer_ci_high - cer_ci_low) / 2
    
    return (wer_mean, wer_half_width), (cer_mean, cer_half_width)

def analyze_csv(filename):
    """Analyze a single CSV file"""
    print(f"\nAnalyzing {filename}...")
    
    # Read CSV
    df = pd.read_csv(filename)
    
    # Get references and predictions
    refs = df['reference'].tolist()
    preds = df['prediction'].tolist()
    
    # Calculate bootstrap statistics
    (wer_mean, wer_half_width), (cer_mean, cer_half_width) = bootstrap_confidence_interval(refs, preds)
    
    # Convert to percentage
    wer_mean_pct = wer_mean * 100
    wer_half_width_pct = wer_half_width * 100
    cer_mean_pct = cer_mean * 100
    cer_half_width_pct = cer_half_width * 100
    
    print(f"{filename}:")
    print(f"  WER: {wer_mean_pct:.2f} ± {wer_half_width_pct:.2f}")
    print(f"  CER: {cer_mean_pct:.2f} ± {cer_half_width_pct:.2f}")
    
    return filename, wer_mean_pct, wer_half_width_pct, cer_mean_pct, cer_half_width_pct

# Main execution
if __name__ == "__main__":
    # csv_files = [
    #     'hviske_coral_results.csv',
    #     'hviske_nst_results.csv', 
    #     'whisper_coral_results.csv',
    #     'whisper_nst_results.csv'
    # ]
    csv_files = ["hviske_librispeech_results.csv"]
    print(f"Using {max(1, os.cpu_count() // 2)} cores for parallel processing")
    
    results = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            result = analyze_csv(csv_file)
            results.append(result)
        else:
            print(f"Warning: {csv_file} not found")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        filename, wer_mean, wer_hw, cer_mean, cer_hw = result
        print(f"{filename:30s}")
        print(f"  WER: {wer_mean:.2f} ± {wer_hw:.2f}%")
        print(f"  CER: {cer_mean:.2f} ± {cer_hw:.2f}%")