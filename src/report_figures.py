import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for academic papers
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Create output directory
os.makedirs('report_figures', exist_ok=True)

# Label mapping
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

def create_class_distribution_comparison():
    """Figure 1: Original vs Balanced Distribution"""
    
    # Data from your output
    original = {
        0: 442096, 1: 4162, 2: 8993, 3: 3029, 4: 2227,
        5: 1968, 6: 5783, 7: 8197, 8: 8127, 9: 4608, 10: 10810
    }
    
    balanced = {
        0: 50000, 1: 4162, 2: 8993, 3: 3029, 4: 2227,
        5: 1968, 6: 5783, 7: 8197, 8: 8127, 9: 4608, 10: 10810
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    labels = [LABEL_NAMES[i] for i in range(11)]
    x_pos = np.arange(len(labels))
    
    # Original distribution
    original_values = [original[i] for i in range(11)]
    bars1 = ax1.bar(x_pos, original_values, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Original Distribution\n(Imbalance Ratio: 224.6:1)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight human class
    bars1[0].set_color('darkred')
    bars1[0].set_alpha(0.9)
    
    # Add value labels for top 3
    for i, v in enumerate(original_values[:3]):
        ax1.text(i, v + 5000, f'{v:,}', ha='center', va='bottom', fontsize=8)
    
    # Balanced distribution
    balanced_values = [balanced[i] for i in range(11)]
    bars2 = ax2.bar(x_pos, balanced_values, color='seagreen', alpha=0.8)
    ax2.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Balanced Distribution\n(Imbalance Ratio: 25.4:1)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight human class
    bars2[0].set_color('darkgreen')
    bars2[0].set_alpha(0.9)
    
    # Add value labels for all bars in balanced
    for i, v in enumerate(balanced_values):
        if v > 5000:
            ax2.text(i, v + 500, f'{v:,}', ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(i, v + 200, f'{v:,}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('report_figures/fig1_class_distribution.png', bbox_inches='tight')
    plt.savefig('report_figures/fig1_class_distribution.pdf', bbox_inches='tight')
    print("✓ Created Figure 1: Class Distribution Comparison")
    plt.close()

def create_imbalance_ratio_comparison():
    """Figure 2: Imbalance Ratio Visualization"""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['Original\nDataset', 'Balanced\nDataset']
    ratios = [224.6, 25.4]
    colors = ['#e74c3c', '#27ae60']
    
    bars = ax.bar(categories, ratios, color=colors, alpha=0.8, width=0.5)
    
    ax.set_ylabel('Imbalance Ratio (majority:minority)', fontsize=11, fontweight='bold')
    ax.set_title('Class Imbalance Reduction\n(88.7% Improvement)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 250)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{ratio:.1f}:1',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement arrow
    ax.annotate('', xy=(1, 25.4), xytext=(0, 224.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black', ls='--'))
    ax.text(0.5, 125, '88.7%\nreduction', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('report_figures/fig2_imbalance_ratio.png', bbox_inches='tight')
    plt.savefig('report_figures/fig2_imbalance_ratio.pdf', bbox_inches='tight')
    print("✓ Created Figure 2: Imbalance Ratio Comparison")
    plt.close()

def create_language_distribution():
    """Figure 3: Programming Language Distribution"""
    
    languages = {
        'Java': 137076,
        'Python': 136709,
        'C#': 62781,
        'JavaScript': 41780,
        'C++': 36581,
        'Go': 29685,
        'PHP': 29198,
        'C': 26190
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    langs = list(languages.keys())
    counts = list(languages.values())
    percentages = [c/sum(counts)*100 for c in counts]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(langs)))
    bars = ax.barh(langs, counts, color=colors, alpha=0.8)
    
    ax.set_xlabel('Sample Count', fontsize=11, fontweight='bold')
    ax.set_ylabel('Programming Language', fontsize=11, fontweight='bold')
    ax.set_title('Programming Language Distribution in Training Set', 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        width = bar.get_width()
        ax.text(width + 2000, bar.get_y() + bar.get_height()/2.,
                f'{count:,} ({pct:.1f}%)',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('report_figures/fig3_language_distribution.png', bbox_inches='tight')
    plt.savefig('report_figures/fig3_language_distribution.pdf', bbox_inches='tight')
    print("✓ Created Figure 3: Language Distribution")
    plt.close()

def create_training_efficiency():
    """Figure 4: Training Efficiency Comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dataset size comparison
    configs = ['Original\nDataset', 'Balanced\nDataset', 'Ultra-Balanced\nDataset']
    sizes = [500000, 107904, 55000]
    colors = ['#e74c3c', '#27ae60', '#3498db']
    
    bars = ax1.bar(configs, sizes, color=colors, alpha=0.8)
    ax1.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Dataset Size Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10000,
                f'{size:,}',
                ha='center', va='bottom', fontsize=9)
    
    # Training time comparison
    times = [14, 3.5, 1.5]  # in hours
    bars2 = ax2.bar(configs, times, color=colors, alpha=0.8)
    ax2.set_ylabel('Training Time (hours)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Expected Training Time\n(RTX 4070 Super, 5 epochs)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{time:.1f}h',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('report_figures/fig4_training_efficiency.png', bbox_inches='tight')
    plt.savefig('report_figures/fig4_training_efficiency.pdf', bbox_inches='tight')
    print("✓ Created Figure 4: Training Efficiency")
    plt.close()

def create_methodology_diagram():
    """Figure 5: Methodology Pipeline"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Define boxes
    boxes = [
        # Input
        {'xy': (0.1, 0.85), 'width': 0.2, 'height': 0.08, 
         'text': 'Raw Code\nSnippets\n(500K samples)', 'color': '#ecf0f1'},
        
        # Preprocessing
        {'xy': (0.1, 0.70), 'width': 0.2, 'height': 0.08,
         'text': 'Data Balancing\n(107K samples)', 'color': '#3498db'},
        
        {'xy': (0.4, 0.70), 'width': 0.2, 'height': 0.08,
         'text': 'Tokenization\n(512 tokens)', 'color': '#3498db'},
        
        # Model
        {'xy': (0.25, 0.52), 'width': 0.25, 'height': 0.08,
         'text': 'CodeBERT\nEncoder\n(125M params)', 'color': '#e74c3c'},
        
        # Loss functions
        {'xy': (0.1, 0.34), 'width': 0.15, 'height': 0.08,
         'text': 'Focal Loss\n(γ=3.0)', 'color': '#f39c12'},
        
        {'xy': (0.3, 0.34), 'width': 0.15, 'height': 0.08,
         'text': 'Weighted\nSampling', 'color': '#f39c12'},
        
        {'xy': (0.5, 0.34), 'width': 0.15, 'height': 0.08,
         'text': 'Class\nWeights', 'color': '#f39c12'},
        
        # Output
        {'xy': (0.25, 0.16), 'width': 0.25, 'height': 0.08,
         'text': '11-Class\nPrediction', 'color': '#27ae60'},
    ]
    
    # Draw boxes
    for box in boxes:
        rect = plt.Rectangle(box['xy'], box['width'], box['height'],
                            facecolor=box['color'], edgecolor='black',
                            linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(box['xy'][0] + box['width']/2, 
               box['xy'][1] + box['height']/2,
               box['text'], ha='center', va='center',
               fontsize=9, fontweight='bold', wrap=True)
    
    # Draw arrows
    arrows = [
        ((0.2, 0.85), (0.2, 0.78)),  # Raw -> Balancing
        ((0.3, 0.74), (0.4, 0.74)),  # Balancing -> Tokenization
        ((0.5, 0.70), (0.375, 0.60)),  # Tokenization -> CodeBERT
        ((0.375, 0.52), (0.175, 0.42)),  # CodeBERT -> Focal
        ((0.375, 0.52), (0.375, 0.42)),  # CodeBERT -> Weighted
        ((0.375, 0.52), (0.575, 0.42)),  # CodeBERT -> Class Weights
        ((0.175, 0.34), (0.3, 0.24)),  # Focal -> Output
        ((0.375, 0.34), (0.375, 0.24)),  # Weighted -> Output
        ((0.575, 0.34), (0.45, 0.24)),  # Class Weights -> Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 1)
    ax.set_title('Training Pipeline: Multi-Level Imbalance Handling', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('report_figures/fig5_methodology_pipeline.png', bbox_inches='tight')
    plt.savefig('report_figures/fig5_methodology_pipeline.pdf', bbox_inches='tight')
    print("✓ Created Figure 5: Methodology Pipeline")
    plt.close()

def create_focal_loss_illustration():
    """Figure 6: Focal Loss Effect"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate probability range
    pt = np.linspace(0.01, 1, 100)
    
    # Calculate losses
    ce_loss = -np.log(pt)
    focal_gamma_1 = -(1 - pt)**1 * np.log(pt)
    focal_gamma_2 = -(1 - pt)**2 * np.log(pt)
    focal_gamma_3 = -(1 - pt)**3 * np.log(pt)
    
    ax.plot(pt, ce_loss, 'k-', linewidth=2, label='Cross-Entropy', alpha=0.7)
    ax.plot(pt, focal_gamma_1, 'b--', linewidth=2, label='Focal Loss (γ=1)', alpha=0.7)
    ax.plot(pt, focal_gamma_2, 'g--', linewidth=2, label='Focal Loss (γ=2)', alpha=0.7)
    ax.plot(pt, focal_gamma_3, 'r-', linewidth=3, label='Focal Loss (γ=3, ours)', alpha=0.9)
    
    ax.set_xlabel('Model Confidence (pt)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=11, fontweight='bold')
    ax.set_title('Focal Loss: Down-weighting Easy Examples', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Highlight easy vs hard examples
    ax.axvspan(0.8, 1.0, alpha=0.1, color='green', label='Easy Examples')
    ax.axvspan(0.0, 0.5, alpha=0.1, color='red', label='Hard Examples')
    
    ax.text(0.9, 4, 'Easy\n(down-weighted)', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(0.25, 4, 'Hard\n(high weight)', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('report_figures/fig6_focal_loss.png', bbox_inches='tight')
    plt.savefig('report_figures/fig6_focal_loss.pdf', bbox_inches='tight')
    print("✓ Created Figure 6: Focal Loss Illustration")
    plt.close()

def main():
    print("="*80)
    print("GENERATING REPORT FIGURES")
    print("="*80)
    print()
    
    create_class_distribution_comparison()
    create_imbalance_ratio_comparison()
    create_language_distribution()
    create_training_efficiency()
    create_methodology_diagram()
    create_focal_loss_illustration()
    
    print()
    print("="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*80)
    print()
    print("Files saved in 'report_figures/' directory:")
    print("  - fig1_class_distribution.png/.pdf")
    print("  - fig2_imbalance_ratio.png/.pdf")
    print("  - fig3_language_distribution.png/.pdf")
    print("  - fig4_training_efficiency.png/.pdf")
    print("  - fig5_methodology_pipeline.png/.pdf")
    print("  - fig6_focal_loss.png/.pdf")
    print()
    print("You can now include these figures in your report!")

if __name__ == '__main__':
    main()