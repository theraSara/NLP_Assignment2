import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

def explore_data(train_file, val_file=None):
    """
    Comprehensive data exploration
    """
    print("="*80)
    print("DATA EXPLORATION")
    print("="*80)
    
    # Load data
    print("\nLoading training data...")
    train_df = pd.read_parquet(train_file)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Columns: {train_df.columns.tolist()}")
    
    # Label distribution
    print("\n" + "-"*80)
    print("LABEL DISTRIBUTION")
    print("-"*80)
    label_counts = train_df['label'].value_counts().sort_index()
    print(label_counts)
    
    # Calculate class imbalance ratio
    max_count = label_counts.max()
    min_count = label_counts.min()
    print(f"\nClass imbalance ratio: {max_count/min_count:.2f}:1")
    
    # Label names
    label_names = {
        0: 'Human', 1: 'DeepSeek-AI', 2: 'Qwen', 3: '01-ai',
        4: 'BigCode', 5: 'Gemma', 6: 'Phi', 7: 'Meta-LLaMA',
        8: 'IBM-Granite', 9: 'Mistral', 10: 'OpenAI'
    }
    
    print("\nDetailed label distribution:")
    for label, count in label_counts.items():
        pct = 100 * count / len(train_df)
        print(f"  {label}: {label_names[label]:15s} - {count:7d} samples ({pct:5.2f}%)")
    
    # Language distribution
    if 'language' in train_df.columns:
        print("\n" + "-"*80)
        print("PROGRAMMING LANGUAGE DISTRIBUTION")
        print("-"*80)
        lang_counts = train_df['language'].value_counts()
        print(lang_counts)
    
    # Generator distribution
    if 'generator' in train_df.columns:
        print("\n" + "-"*80)
        print("GENERATOR DISTRIBUTION")
        print("-"*80)
        gen_counts = train_df['generator'].value_counts()
        print(gen_counts.head(20))
        print(f"... and {len(gen_counts) - 20} more generators")
    
    # Code length statistics
    print("\n" + "-"*80)
    print("CODE LENGTH STATISTICS")
    print("-"*80)
    train_df['code_length'] = train_df['code'].str.len()
    print(train_df['code_length'].describe())
    
    # Validation data
    if val_file:
        print("\n" + "="*80)
        print("VALIDATION DATA")
        print("="*80)
        val_df = pd.read_parquet(val_file)
        print(f"Validation samples: {len(val_df)}")
        print("\nLabel distribution:")
        val_label_counts = val_df['label'].value_counts().sort_index()
        for label, count in val_label_counts.items():
            pct = 100 * count / len(val_df)
            print(f"  {label}: {label_names[label]:15s} - {count:7d} samples ({pct:5.2f}%)")
    
    print("\n" + "="*80)
    
    return train_df

def calculate_class_weights(train_file):
    """
    Calculate class weights for imbalanced dataset
    """
    train_df = pd.read_parquet(train_file)
    labels = train_df['label'].values
    
    class_counts = Counter(labels)
    total = len(labels)
    
    # Method 1: Inverse frequency
    weights_inv_freq = {cls: total / (len(class_counts) * count) 
                        for cls, count in class_counts.items()}
    
    # Method 2: Effective number of samples
    beta = 0.9999
    weights_ens = {cls: (1 - beta) / (1 - beta**count)
                   for cls, count in class_counts.items()}
    
    print("Class Weights (Inverse Frequency):")
    for cls in sorted(weights_inv_freq.keys()):
        print(f"  Class {cls}: {weights_inv_freq[cls]:.4f}")
    
    print("\nClass Weights (Effective Number of Samples):")
    for cls in sorted(weights_ens.keys()):
        print(f"  Class {cls}: {weights_ens[cls]:.4f}")
    
    return weights_inv_freq, weights_ens

def sample_by_class(train_file, output_file, samples_per_class=5000):
    """
    Create a balanced subset by sampling from each class
    """
    train_df = pd.read_parquet(train_file)
    
    sampled_dfs = []
    for label in range(11):
        class_df = train_df[train_df['label'] == label]
        n_samples = min(samples_per_class, len(class_df))
        sampled = class_df.sample(n=n_samples, random_state=42)
        sampled_dfs.append(sampled)
    
    balanced_df = pd.concat(sampled_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    balanced_df.to_parquet(output_file)
    print(f"Balanced dataset saved to {output_file}")
    print(f"Total samples: {len(balanced_df)}")
    print("\nLabel distribution:")
    print(balanced_df['label'].value_counts().sort_index())
    
    return balanced_df

def visualize_training_history(history_file, output_file='training_history.png'):
    """
    Visualize training history
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_f1 = [h['train_f1'] for h in history]
    val_loss = [h.get('val_loss', 0) for h in history]
    val_f1 = [h['val_f1'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', marker='o')
    if any(val_loss):
        ax1.plot(epochs, val_loss, 'r-', label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 plot
    ax2.plot(epochs, train_f1, 'b-', label='Train F1', marker='o')
    ax2.plot(epochs, val_f1, 'r-', label='Val F1', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1 Score')
    ax2.set_title('Training and Validation F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {output_file}")
    plt.close()

def check_submission_format(submission_file):
    """
    Validate submission file format
    """
    print("Checking submission format...")
    
    # Load submission
    df = pd.read_csv(submission_file)
    
    # Check columns
    required_columns = ['id', 'label']
    if not all(col in df.columns for col in required_columns):
        print(f"ERROR: Missing required columns. Found: {df.columns.tolist()}")
        return False
    
    print("✓ Columns are correct")
    
    # Check id column
    if df['id'].dtype not in [np.int32, np.int64]:
        print(f"ERROR: 'id' column should be integer, found: {df['id'].dtype}")
        return False
    
    print("✓ ID column type is correct")
    
    # Check label column
    if df['label'].dtype not in [np.int32, np.int64]:
        print(f"ERROR: 'label' column should be integer, found: {df['label'].dtype}")
        return False
    
    print("✓ Label column type is correct")
    
    # Check label range
    valid_labels = set(range(11))
    unique_labels = set(df['label'].unique())
    
    if not unique_labels.issubset(valid_labels):
        invalid = unique_labels - valid_labels
        print(f"ERROR: Found invalid labels: {invalid}")
        return False
    
    print(f"All labels are valid (0-10)")
    
    # Check for duplicates
    if df['id'].duplicated().any():
        print(f"ERROR: Found duplicate IDs")
        return False
    
    print("✓ No duplicate IDs")
    
    # Check for missing values
    if df.isnull().any().any():
        print(f"ERROR: Found missing values")
        return False
    
    print("No missing values")
    
    # Print summary
    print(f"\nSubmission format is valid!")
    print(f"  - Total predictions: {len(df)}")
    print(f"  - Label distribution:")
    for label, count in df['label'].value_counts().sort_index().items():
        pct = 100 * count / len(df)
        print(f"      {label}: {count:6d} ({pct:5.2f}%)")
    
    return True

def compare_predictions(pred_file1, pred_file2, ground_truth_file=None):
    """
    Compare two prediction files
    """
    print("Comparing predictions...")
    
    pred1 = pd.read_csv(pred_file1)
    pred2 = pd.read_csv(pred_file2)
    
    # Check agreement
    agreement = (pred1['label'] == pred2['label']).mean()
    print(f"\nAgreement between predictions: {agreement:.3f}")
    
    # Where they differ
    diff_mask = pred1['label'] != pred2['label']
    diff_count = diff_mask.sum()
    
    print(f"Number of differences: {diff_count} ({100*diff_count/len(pred1):.2f}%)")
    
    if ground_truth_file:
        # Load ground truth
        if ground_truth_file.endswith('.parquet'):
            gt_df = pd.read_parquet(ground_truth_file)
        else:
            gt_df = pd.read_csv(ground_truth_file)
        
        gt = gt_df['label'].values
        
        # Calculate accuracies
        acc1 = (pred1['label'] == gt).mean()
        acc2 = (pred2['label'] == gt).mean()
        
        print(f"\nAccuracy of prediction 1: {acc1:.4f}")
        print(f"Accuracy of prediction 2: {acc2:.4f}")
        
        # Where pred1 is correct but pred2 is wrong
        pred1_right_pred2_wrong = ((pred1['label'] == gt) & (pred2['label'] != gt)).sum()
        pred2_right_pred1_wrong = ((pred2['label'] == gt) & (pred1['label'] != gt)).sum()
        
        print(f"\nPred1 correct but Pred2 wrong: {pred1_right_pred2_wrong}")
        print(f"Pred2 correct but Pred1 wrong: {pred2_right_pred1_wrong}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python utils.py explore <train_file> [val_file]")
        print("  python utils.py weights <train_file>")
        print("  python utils.py sample <train_file> <output_file> [samples_per_class]")
        print("  python utils.py check <submission_file>")
        print("  python utils.py compare <pred1> <pred2> [ground_truth]")
        print("  python utils.py visualize <history_json> [output_png]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'explore':
        explore_data(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    
    elif command == 'weights':
        calculate_class_weights(sys.argv[2])
    
    elif command == 'sample':
        samples = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
        sample_by_class(sys.argv[2], sys.argv[3], samples)
    
    elif command == 'check':
        check_submission_format(sys.argv[2])
    
    elif command == 'compare':
        gt = sys.argv[4] if len(sys.argv) > 4 else None
        compare_predictions(sys.argv[2], sys.argv[3], gt)
    
    elif command == 'visualize':
        output = sys.argv[3] if len(sys.argv) > 3 else 'training_history.png'
        visualize_training_history(sys.argv[2], output)
    
    else:
        print(f"Unknown command: {command}")