import pandas as pd
import numpy as np
import argparse
from collections import Counter

def create_balanced_dataset(train_file, output_file, 
                           human_samples=50000, 
                           oversample_minority=False,
                           target_per_class=None,
                           random_state=42):
    
    # Load data
    print(f"\nLoading data from: {train_file}")
    train_df = pd.read_parquet(train_file)
    
    print(f"Original dataset size: {len(train_df)}")
    print("\nOriginal label distribution:")
    original_dist = train_df['label'].value_counts().sort_index()
    print(original_dist)
    
    balanced_dfs = []
    
    # Handle Human class (label 0)
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
