import matplotlib.pyplot as plt
import numpy as np

# Set style for academic papers
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Generate probability range
pt = np.linspace(0.01, 1, 200)

# Calculate losses for different gamma values
ce_loss = -np.log(pt)
focal_gamma_1 = -(1 - pt)**1 * np.log(pt)
focal_gamma_2 = -(1 - pt)**2 * np.log(pt)
focal_gamma_3 = -(1 - pt)**3 * np.log(pt)

# Create figure
fig, ax = plt.subplots()

# Plot lines
ax.plot(pt, ce_loss, 'k-', linewidth=2, label='Cross-Entropy', alpha=0.7)
ax.plot(pt, focal_gamma_1, 'b--', linewidth=1.5, label='Focal ($\\gamma=1$)', alpha=0.7)
ax.plot(pt, focal_gamma_2, 'g--', linewidth=1.5, label='Focal ($\\gamma=2$)', alpha=0.7)
ax.plot(pt, focal_gamma_3, 'r-', linewidth=2.5, label='Focal ($\\gamma=3$, ours)', alpha=0.9)

# Labels and title
ax.set_xlabel('Model Confidence ($p_t$)', fontweight='bold')
ax.set_ylabel('Loss Value', fontweight='bold')
ax.set_title('Focal Loss: Down-weighting Well-Classified Examples', fontweight='bold', fontsize=11)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Add annotations
ax.axvspan(0.8, 1.0, alpha=0.1, color='green')
ax.axvspan(0.0, 0.5, alpha=0.1, color='red')

ax.text(0.9, 3.5, 'Easy\n(down-weighted)', ha='center', fontsize=8,
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6, edgecolor='darkgreen'))
ax.text(0.25, 3.5, 'Hard\n(high weight)', ha='center', fontsize=8,
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6, edgecolor='darkred'))

# Set limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 5)

plt.tight_layout()

# Save in both formats
plt.savefig('focal_loss_figure.pdf', bbox_inches='tight', dpi=300)
plt.savefig('focal_loss_figure.png', bbox_inches='tight', dpi=300)

print("✓ Generated focal_loss_figure.pdf and focal_loss_figure.png")
print("✓ Upload the PDF to Overleaf and include in your report!")

plt.close()