import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import json
import argparse
import os

# Label mapping for Subtask B
LABEL_NAMES = {
    0: 'Human',
    1: 'DeepSeek-AI',
    2: 'Qwen',
    3: '01-ai',
    4: 'BigCode',
    5: 'Gemma',
    6: 'Phi',
    7: 'Meta-LLaMA',
    8: 'IBM-Granite',
    9: 'Mistral',
    10: 'OpenAI'
}

def load_predictions(pred_file, label_col='label'):
    df = pd.read_csv(pred_file)
    return df[label_col].values

def load_ground_truth(data_file):
    df = pd.read_parquet(data_file)
    return df['label'].values, df

def plot_confusion_matrix(y_true, y_pred, output_dir, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=range(11))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', square=True,
                xticklabels=[LABEL_NAMES[i] for i in range(11)],
                yticklabels=[LABEL_NAMES[i] for i in range(11)])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    suffix = '_normalized' if normalize else ''
    plt.savefig(os.path.join(output_dir, f'confusion_matrix{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_f1(y_true, y_pred, output_dir):
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(11))
    
    plt.figure(figsize=(12, 6))
    x_labels = [LABEL_NAMES[i] for i in range(11)]
    x_pos = np.arange(len(x_labels))
    
    colors = ['#2ecc71' if f1 > 0.5 else '#e74c3c' for f1 in f1_per_class]
    plt.bar(x_pos, f1_per_class, color=colors, alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores')
    plt.xticks(x_pos, x_labels, rotation=45, ha='right')
    plt.axhline(y=np.mean(f1_per_class), color='blue', linestyle='--', 
                label=f'Macro F1: {np.mean(f1_per_class):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return f1_per_class

def analyze_errors_by_language(y_true, y_pred, df, output_dir):
    if 'language' not in df.columns:
        print("No language information available")
        return
    
    df_analysis = df.copy()
    df_analysis['predicted'] = y_pred
    df_analysis['correct'] = (y_true == y_pred).astype(int)
    
    # Accuracy by language
    lang_acc = df_analysis.groupby('language')['correct'].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    lang_acc.plot(kind='barh', color='skyblue')
    plt.xlabel('Accuracy')
    plt.ylabel('Programming Language')
    plt.title('Accuracy by Programming Language')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_language.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed stats
    lang_stats = df_analysis.groupby('language').agg({
        'correct': ['mean', 'count']
    }).round(3)
    lang_stats.columns = ['accuracy', 'count']
    lang_stats.to_csv(os.path.join(output_dir, 'language_stats.csv'))
    
    return lang_stats

def analyze_errors_by_generator(y_true, y_pred, df, output_dir):
    """Analyze errors by generator"""
    if 'generator' not in df.columns:
        print("No generator information available")
        return
    
    df_analysis = df.copy()
    df_analysis['predicted'] = y_pred
    df_analysis['correct'] = (y_true == y_pred).astype(int)
    
    # Accuracy by generator
    gen_acc = df_analysis.groupby('generator')['correct'].mean().sort_values()
    
    plt.figure(figsize=(12, 8))
    gen_acc.plot(kind='barh', color='lightcoral')
    plt.xlabel('Accuracy')
    plt.ylabel('Generator')
    plt.title('Accuracy by Generator')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_generator.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed stats
    gen_stats = df_analysis.groupby('generator').agg({
        'correct': ['mean', 'count']
    }).round(3)
    gen_stats.columns = ['accuracy', 'count']
    gen_stats.to_csv(os.path.join(output_dir, 'generator_stats.csv'))
    
    return gen_stats

def analyze_class_distribution(y_true, y_pred, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # True distribution
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    
    x_labels = [LABEL_NAMES[i] for i in range(11)]
    x_pos = np.arange(len(x_labels))
    
    ax1.bar(x_pos, [true_counts.get(i, 0) for i in range(11)], alpha=0.7, color='green')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('True Label Distribution')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    
    ax2.bar(x_pos, [pred_counts.get(i, 0) for i in range(11)], alpha=0.7, color='blue')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_title('Predicted Label Distribution')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(y_true, y_pred, output_dir):
    report = classification_report(
        y_true, y_pred, 
        target_names=[LABEL_NAMES[i] for i in range(11)],
        zero_division=0,
        digits=3
    )
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(report)
    
    # Save to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Calculate and save macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'macro_f1': float(macro_f1),
        'accuracy': float((y_true == y_pred).mean())
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Analyze predictions for Subtask B')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions CSV file')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth parquet file')
    parser.add_argument('--output_dir', type=str, default='analysis',
                        help='Output directory for plots and reports')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading predictions and ground truth...")
    y_pred = load_predictions(args.predictions)
    y_true, df = load_ground_truth(args.ground_truth)
    
    print(f"Loaded {len(y_true)} samples")
    
    # Generate comprehensive analysis
    print("\nGenerating classification report...")
    metrics = generate_report(y_true, y_pred, args.output_dir)
    
    print("\nPlotting confusion matrices...")
    plot_confusion_matrix(y_true, y_pred, args.output_dir, normalize=False)
    plot_confusion_matrix(y_true, y_pred, args.output_dir, normalize=True)
    
    print("\nPlotting per-class F1 scores...")
    f1_per_class = plot_per_class_f1(y_true, y_pred, args.output_dir)
    
    print("\nAnalyzing class distribution...")
    analyze_class_distribution(y_true, y_pred, args.output_dir)
    
    print("\nAnalyzing errors by language...")
    lang_stats = analyze_errors_by_language(y_true, y_pred, df, args.output_dir)
    
    print("\nAnalyzing errors by generator...")
    gen_stats = analyze_errors_by_generator(y_true, y_pred, df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Print per-class F1 scores
    print("\nPer-class F1 Scores:")
    for i, f1 in enumerate(f1_per_class):
        print(f"  {LABEL_NAMES[i]:15s}: {f1:.3f}")

if __name__ == '__main__':
    main()