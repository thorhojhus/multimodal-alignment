import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import Audio

# Import datasets
from dataset.load_coral import coral_dataset_dict
from dataset.load_librispeech import librispeech_dataset_dict
from dataset.load_melatech import melatech_dataset_dict
from dataset.nst_da import nst_dataset_dict

def analyze_dataset(dataset, name, split):
    """Analyze audio length statistics for a dataset."""
    print(f"Analyzing {name} - {split}")
    
    # Ensure audio column is properly loaded
    if 'audio' not in dataset.column_names:
        print(f"  No 'audio' column in {name} - {split}")
        return None
    
    # Check if audio needs casting
    if not isinstance(dataset[0]['audio'], dict) or 'array' not in dataset[0]['audio']:
        try:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        except Exception as e:
            print(f"  Error casting audio column: {e}")
            return None
    
    # Collect lengths in seconds and samples
    lengths_seconds = []
    lengths_samples = []
    
    for item in tqdm(dataset, desc=f"{name}-{split}", total=len(dataset)):
        try:
            audio = item['audio']
            if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
                samples = len(audio['array'])
                seconds = samples / audio['sampling_rate']
                lengths_seconds.append(seconds)
                lengths_samples.append(samples)
        except Exception as e:
            print(f"  Error processing audio item: {e}")
    
    if not lengths_seconds:
        print(f"  No valid audio found in {name} - {split}")
        return None
    
    stats = {
        'name': name,
        'split': split,
        'count': len(lengths_seconds),
        'min_seconds': min(lengths_seconds),
        'max_seconds': max(lengths_seconds),
        'mean_seconds': np.mean(lengths_seconds),
        'median_seconds': np.median(lengths_seconds),
        'p95_seconds': np.percentile(lengths_seconds, 95),
        'p99_seconds': np.percentile(lengths_seconds, 99),
        'min_samples': min(lengths_samples),
        'max_samples': max(lengths_samples),
        'mean_samples': np.mean(lengths_samples),
        'median_samples': np.median(lengths_samples),
        'p95_samples': np.percentile(lengths_samples, 95),
        'p99_samples': np.percentile(lengths_samples, 99),
    }
    
    print(f"  Statistics for {name} - {split}:")
    print(f"  Count: {stats['count']}")
    print(f"  Duration (seconds): min={stats['min_seconds']:.2f}, max={stats['max_seconds']:.2f}, mean={stats['mean_seconds']:.2f}, median={stats['median_seconds']:.2f}")
    print(f"  Duration (samples@16kHz): min={stats['min_samples']}, max={stats['max_samples']}, mean={stats['mean_samples']:.1f}, median={stats['median_samples']:.1f}")
    print(f"  95th percentile (seconds): {stats['p95_seconds']:.2f}")
    print(f"  99th percentile (seconds): {stats['p99_seconds']:.2f}")
    print(f"  95th percentile (samples): {stats['p95_samples']:.1f}")
    print(f"  99th percentile (samples): {stats['p99_samples']:.1f}")
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths_seconds, bins=50)
    plt.title(f'Audio Duration Distribution - {name} - {split}')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.axvline(stats['mean_seconds'], color='r', linestyle='--', label=f'Mean: {stats["mean_seconds"]:.2f}s')
    plt.axvline(stats['median_seconds'], color='g', linestyle='--', label=f'Median: {stats["median_seconds"]:.2f}s')
    plt.axvline(stats['p95_seconds'], color='m', linestyle='--', label=f'95th: {stats["p95_seconds"]:.2f}s')
    plt.legend()
    
    # Create output directory
    os.makedirs('audio_analysis', exist_ok=True)
    plt.savefig(f'audio_analysis/{name}_{split}_histogram.png')
    plt.close()
    
    return stats

def main():
    all_stats = []
    
    # Coral dataset
    for split in coral_dataset_dict:
        stats = analyze_dataset(coral_dataset_dict[split], "coral", split)
        if stats:
            all_stats.append(stats)
    
    # LibriSpeech dataset
    for split in librispeech_dataset_dict:
        stats = analyze_dataset(librispeech_dataset_dict[split], "librispeech", split)
        if stats:
            all_stats.append(stats)
    
    # NST Da
    for split in nst_dataset_dict:
        stats = analyze_dataset(nst_dataset_dict[split], "NST-Da", split)
        if stats:
            all_stats.append(stats)
    
    # Melatech dataset
    for split in melatech_dataset_dict:
        stats = analyze_dataset(melatech_dataset_dict[split], "melatech", split)
        if stats:
            all_stats.append(stats)
    
    # Write summary to file
    with open('audio_analysis/summary.txt', 'w') as f:
        f.write("# Audio Length Analysis Summary\n\n")
        f.write("| Dataset | Split | Count | Min (s) | Max (s) | Mean (s) | Median (s) | P95 (s) | P99 (s) | Max (samples) | P95 (samples) |\n")
        f.write("|---------|-------|-------|---------|---------|----------|------------|---------|---------|--------------|---------------|\n")
        
        for stats in all_stats:
            f.write(f"| {stats['name']} | {stats['split']} | {stats['count']} | {stats['min_seconds']:.2f} | {stats['max_seconds']:.2f} | {stats['mean_seconds']:.2f} | {stats['median_seconds']:.2f} | {stats['p95_seconds']:.2f} | {stats['p99_seconds']:.2f} | {stats['max_samples']} | {stats['p95_samples']:.0f} |\n")
    
    # Find maximum values across all datasets
    if all_stats:
        max_seconds = max(s['max_seconds'] for s in all_stats)
        p95_seconds = max(s['p95_seconds'] for s in all_stats)
        p99_seconds = max(s['p99_seconds'] for s in all_stats)
        max_samples = max(s['max_samples'] for s in all_stats)
        p95_samples = max(s['p95_samples'] for s in all_stats)
        p99_samples = max(s['p99_samples'] for s in all_stats)
        
        print("\n\n===== MAXIMUM VALUES ACROSS ALL DATASETS =====")
        print(f"Max seconds: {max_seconds:.2f}")
        print(f"95th percentile seconds: {p95_seconds:.2f}")
        print(f"99th percentile seconds: {p99_seconds:.2f}")
        print(f"Max samples: {max_samples}")
        print(f"95th percentile samples: {p95_samples:.0f}")
        print(f"99th percentile samples: {p99_samples:.0f}")
        
        with open('audio_analysis/summary.txt', 'a') as f:
            f.write("\n\n## Maximum Values Across All Datasets\n\n")
            f.write(f"- Max seconds: {max_seconds:.2f}\n")
            f.write(f"- 95th percentile seconds: {p95_seconds:.2f}\n")
            f.write(f"- 99th percentile seconds: {p99_seconds:.2f}\n")
            f.write(f"- Max samples: {max_samples}\n")
            f.write(f"- 95th percentile samples: {p95_samples:.0f}\n")
            f.write(f"- 99th percentile samples: {p99_samples:.0f}\n")

if __name__ == "__main__":
    main()