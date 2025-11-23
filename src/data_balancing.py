import pandas as pd
import numpy as np
import argparse
from collections import Counter

def create_balanced_dataset(train_file, output_file, 
                           human_samples=50000, 
                           oversample_minority=False,
                           target_per_class=None,
                           random_state=42):
    """
    Create a balanced dataset for training
    
    Args:
        train_file: Path to training parquet file
        output_file: Path to save balanced dataset
        human_samples: Number of human samples to keep (default: 50000)
        oversample_minority: Whether to oversample minority classes
        target_per_class: Target number of samples per minority class
        random_state: Random seed
    """
    print("="*80)
    print("CREATING BALANCED DATASET")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from: {train_file}")
    train_df = pd.read_parquet(train_file)
    
    print(f"Original dataset size: {len(train_df)}")
    print("\nOriginal label distribution:")
    original_dist = train_df['label'].value_counts().sort_index()
    print(original_dist)
    
    # Calculate class distribution
    label_counts = dict(train_df['label'].value_counts())
    
    balanced_dfs = []
    
    # Handle Human class (label 0)
    print(f"\n{'='*80}")
    print("PROCESSING HUMAN CLASS (Label 0)")
    print(f"{'='*80}")
    human_df = train_df[train_df['label'] == 0]
    original_human = len(human_df)
    
    if human_samples < len(human_df):
        human_sampled = human_df.sample(n=human_samples, random_state=random_state)
        print(f"Undersampling: {original_human} -> {human_samples} samples")
    else:
        human_sampled = human_df
        print(f"Keeping all {original_human} samples (requested {human_samples})")
    
    balanced_dfs.append(human_sampled)
    
    # Handle minority classes (labels 1-10)
    print(f"\n{'='*80}")
    print("PROCESSING MINORITY CLASSES (Labels 1-10)")
    print(f"{'='*80}")
    
    for label in range(1, 11):
        llm_df = train_df[train_df['label'] == label]
        original_count = len(llm_df)
        
        label_names = {
            1: 'DeepSeek-AI', 2: 'Qwen', 3: '01-ai', 4: 'BigCode',
            5: 'Gemma', 6: 'Phi', 7: 'Meta-LLaMA', 8: 'IBM-Granite',
            9: 'Mistral', 10: 'OpenAI'
        }
        
        if oversample_minority and target_per_class and original_count < target_per_class:
            # Oversample to reach target
            n_needed = target_per_class
            sampled = llm_df.sample(n=n_needed, replace=True, random_state=random_state)
            print(f"Label {label} ({label_names[label]:15s}): {original_count:6d} -> {n_needed:6d} (oversampled)")
        else:
            # Keep all samples
            sampled = llm_df
            print(f"Label {label} ({label_names[label]:15s}): {original_count:6d} samples (kept all)")
        
        balanced_dfs.append(sampled)
    
    # Combine all dataframes
    print(f"\n{'='*80}")
    print("COMBINING ALL CLASSES")
    print(f"{'='*80}")
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save
    balanced_df.to_parquet(output_file)
    
    print(f"\nBalanced dataset saved to: {output_file}")
    print(f"Total samples: {len(balanced_df)}")
    
    print("\nNew label distribution:")
    new_dist = balanced_df['label'].value_counts().sort_index()
    print(new_dist)
    
    # Calculate statistics
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    
    print(f"Original size: {len(train_df):,}")
    print(f"Balanced size: {len(balanced_df):,}")
    print(f"Reduction: {100 * (1 - len(balanced_df)/len(train_df)):.1f}%")
    
    # Class imbalance ratio
    max_samples = new_dist.max()
    min_samples = new_dist.min()
    print(f"\nOriginal imbalance ratio: {original_dist.max() / original_dist.min():.1f}:1")
    print(f"New imbalance ratio: {max_samples / min_samples:.1f}:1")
    
    # Calculate expected training time reduction
    time_reduction = 100 * (1 - len(balanced_df) / len(train_df))
    print(f"\nExpected training time reduction: ~{time_reduction:.0f}%")
    
    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")
    
    return balanced_df

def create_ultra_balanced(train_file, output_file, samples_per_class=5000, random_state=42):
    """
    Create an ultra-balanced dataset with equal samples per class
    """
    print("="*80)
    print("CREATING ULTRA-BALANCED DATASET")
    print(f"Target: {samples_per_class} samples per class")
    print("="*80)
    
    train_df = pd.read_parquet(train_file)
    
    balanced_dfs = []
    
    for label in range(11):
        class_df = train_df[train_df['label'] == label]
        original_count = len(class_df)
        
        if original_count >= samples_per_class:
            # Undersample
            sampled = class_df.sample(n=samples_per_class, random_state=random_state)
            print(f"Label {label}: {original_count:6d} -> {samples_per_class:6d} (undersampled)")
        else:
            # Oversample with replacement
            sampled = class_df.sample(n=samples_per_class, replace=True, random_state=random_state)
            print(f"Label {label}: {original_count:6d} -> {samples_per_class:6d} (oversampled)")
        
        balanced_dfs.append(sampled)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    balanced_df.to_parquet(output_file)
    
    print(f"\nUltra-balanced dataset saved to: {output_file}")
    print(f"Total samples: {len(balanced_df)}")
    print("\nLabel distribution:")
    print(balanced_df['label'].value_counts().sort_index())
    
    return balanced_df

def main():
    parser = argparse.ArgumentParser(description='Create balanced dataset for training')
    parser.add_argument('--train_file', type=str, default=r'C:\Users\PC\OneDrive\Documents\uni\NLP_Assignment2\Task_B\train.parquet',
                        help='Input training file')
    parser.add_argument('--output_file', type=str, default=r'C:\Users\PC\OneDrive\Documents\uni\NLP_Assignment2\Task_B\train_balanced.parquet',
                        help='Output balanced file')
    parser.add_argument('--human_samples', type=int, default=50000,
                        help='Number of human samples to keep')
    parser.add_argument('--oversample', action='store_true',
                        help='Oversample minority classes')
    parser.add_argument('--target_per_class', type=int, default=None,
                        help='Target samples per minority class (requires --oversample)')
    parser.add_argument('--ultra_balanced', action='store_true',
                        help='Create ultra-balanced dataset with equal samples per class')
    parser.add_argument('--samples_per_class', type=int, default=5000,
                        help='Samples per class for ultra-balanced (default: 5000)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if args.ultra_balanced:
        create_ultra_balanced(
            args.train_file, 
            args.output_file, 
            args.samples_per_class,
            args.random_state
        )
    else:
        create_balanced_dataset(
            args.train_file,
            args.output_file,
            args.human_samples,
            args.oversample,
            args.target_per_class,
            args.random_state
        )

if __name__ == '__main__':
    main()
    
    
    
    
    
"""
OUTPUT:
CREATING BALANCED DATASET
================================================================================

Loading data from: C:\Users\PC\OneDrive\Documents\uni\NLP_Assignment2\Task_B\train.parquet
Original dataset size: 500000

Original label distribution:
label
0     442096
1       4162
2       8993
3       3029
4       2227
5       1968
6       5783
7       8197
8       8127
9       4608
10     10810
Name: count, dtype: int64

================================================================================
PROCESSING HUMAN CLASS (Label 0)
================================================================================
Undersampling: 442096 -> 50000 samples

================================================================================
PROCESSING MINORITY CLASSES (Labels 1-10)
================================================================================
Label 1 (DeepSeek-AI    ):   4162 samples (kept all)
Label 2 (Qwen           ):   8993 samples (kept all)
Label 3 (01-ai          ):   3029 samples (kept all)
Label 4 (BigCode        ):   2227 samples (kept all)
Label 5 (Gemma          ):   1968 samples (kept all)
Label 6 (Phi            ):   5783 samples (kept all)
Label 7 (Meta-LLaMA     ):   8197 samples (kept all)
Label 8 (IBM-Granite    ):   8127 samples (kept all)
Label 9 (Mistral        ):   4608 samples (kept all)
Label 10 (OpenAI         ):  10810 samples (kept all)

================================================================================
COMBINING ALL CLASSES
================================================================================

Balanced dataset saved to: C:\Users\PC\OneDrive\Documents\uni\NLP_Assignment2\Task_B\train_balanced.parquet
Total samples: 107904

New label distribution:
label
0     50000
1      4162
2      8993
3      3029
4      2227
5      1968
6      5783
7      8197
8      8127
9      4608
10    10810
Name: count, dtype: int64

================================================================================
STATISTICS
================================================================================
Original size: 500,000
Balanced size: 107,904
Reduction: 78.4%

Original imbalance ratio: 224.6:1
New imbalance ratio: 25.4:1

Expected training time reduction: ~78%

================================================================================
DONE!
"""