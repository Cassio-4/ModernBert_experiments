import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual F1 scores)
models = [
    "AngularLoss_NoMiner",
    "ArcFaceLoss_NoMiner",
    "ContrastiveLoss_NoMiner",
    "MultiSimilarityLoss_NoMiner",
    "NPairsLoss_NoMiner",
    "ProxyNCALoss_NoMiner",
    "SignalToNoiseRatioLoss_NoMiner",
    "SoftTripleLoss_NoMiner",
    "SphereFaceLoss_NoMiner",
    "TripletMarginLoss_NoMiner"
]
f1_scores_K1 = [0.820, 0.894, 0.802, 0.887, 0.797,
                0.900, 0.849, 0.909, 0.762, 0.873]
f1_scores_K3 = [0.813, 0.900, 0.797, 0.891, 0.814,
                0.901, 0.847, 0.910, 0.768, 0.872]
# Define colors similar to the paper (green and orange)
colors = ['#4CAF50', '#FF9800']  # Green and orange

# Create horizontal bar positions
y = np.arange(len(models))  # Label locations
height = 0.35  # Bar height

# Plot horizontal bars
fig, ax = plt.subplots(figsize=(10, 6))
bars_with_reg = ax.barh(y - height/2, f1_scores_K1, height, label='K = 1', color=colors[0])
bars_without_reg = ax.barh(y + height/2, f1_scores_K3, height, label='K = 3', color=colors[1])

# Add labels and title
ax.set_xlabel('F1 Score', fontsize=12)
ax.set_ylabel('Experiments', fontsize=12)
ax.set_title('F1 Score for NER with Deep Metric Learning on BERTimbau', fontsize=14)
ax.set_yticks(y)
ax.set_yticklabels(models)
ax.legend(loc='lower right')

# Add value labels on the bars
def add_value_labels(bars):
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.2f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),  # Offset for text
                    textcoords="offset points",
                    ha='left', va='center')

add_value_labels(bars_with_reg)
add_value_labels(bars_without_reg)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig('f1_plot_bertimbau.pdf')

# Close the plot to prevent it from being displayed (optional)
plt.close() 