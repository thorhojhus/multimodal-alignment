import pandas as pd
import numpy as np
from jiwer import wer
from tqdm import tqdm
from joblib import Parallel, delayed
import os

def calculate_single_bootstrap(refs, preds, indices):
    """Calculate WER for a single bootstrap sample"""
    # Sample with replacement
    sample_indices = np.random.choice(indices, size=len(indices), replace=True)
    
    # Get sampled references and predictions
    sampled_refs = [refs[idx] for idx in sample_indices]
    sampled_preds = [preds[idx] for idx in sample_indices]
    
    # Calculate WER on sampled data
    return wer(sampled_refs, sampled_preds)

def parallel_bootstrap_confidence_interval(refs, preds, n_bootstrap=1000, confidence=0.95, n_jobs=-1):
    """
    Calculate bootstrap confidence intervals for WER using parallel processing
    """
    indices = np.arange(len(refs))
    
    # Run bootstrap iterations in parallel
    print(f"Running {n_bootstrap} bootstrap iterations...")
    wer_bootstrap = Parallel(n_jobs=n_jobs)(
        delayed(calculate_single_bootstrap)(refs, preds, indices) 
        for _ in tqdm(range(n_bootstrap))
    )
    
    wer_bootstrap = np.array(wer_bootstrap)
    
    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    
    wer_mean = np.mean(wer_bootstrap)
    wer_ci_low = np.percentile(wer_bootstrap, 100 * alpha)
    wer_ci_high = np.percentile(wer_bootstrap, 100 * (1 - alpha))
    
    return wer_mean, wer_ci_low, wer_ci_high

def analyze_csv_file(csv_path, n_bootstrap=1000):
    """
    Analyze a single CSV file and return bootstrap results
    """
    print(f"\nAnalyzing: {os.path.basename(csv_path)}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Get references and predictions
    refs = df['reference'].tolist()
    preds = df['prediction'].tolist()
    
    # Calculate bootstrap confidence intervals
    wer_mean, wer_ci_low, wer_ci_high = parallel_bootstrap_confidence_interval(
        refs, preds, n_bootstrap=n_bootstrap
    )
    
    # Calculate half-width of confidence interval
    half_width = (wer_ci_high - wer_ci_low) / 2
    
    # Format as "mean ± half_width"
    result_str = f"{wer_mean:.3f} ± {half_width:.3f}"
    
    return {
        'file': os.path.basename(csv_path),
        'wer_mean': wer_mean,
        'wer_ci_low': wer_ci_low,
        'wer_ci_high': wer_ci_high,
        'half_width': half_width,
        'formatted': result_str
    }

def main():
    # List of CSV files to analyze
    csv_files = [
        'hviske_coral_results.csv',
        'hviske_nst_results.csv',
        'whisper_coral_results.csv',
        'whisper_nst_results.csv'
    ]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    results = []
    
    # Analyze each CSV file
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            result = analyze_csv_file(csv_file, n_bootstrap=1000)
            results.append(result)
        else:
            print(f"Warning: {csv_file} not found")
    
    # Print summary
    print("\n" + "="*60)
    print("BOOTSTRAP WER RESULTS (95% Confidence Intervals)")
    print("="*60)
    
    for result in results:
        print(f"{result['file']:<30} WER: {result['formatted']}")
    
    print("\nDetailed results:")
    for result in results:
        print(f"\n{result['file']}:")
        print(f"  Mean WER: {result['wer_mean']:.4f}")
        print(f"  95% CI: [{result['wer_ci_low']:.4f}, {result['wer_ci_high']:.4f}]")
        print(f"  Half-width: {result['half_width']:.4f}")

if __name__ == "__main__":
    main()