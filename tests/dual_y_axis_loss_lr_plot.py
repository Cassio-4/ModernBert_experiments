import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read your CSV file
df = pd.read_csv('/home/ubuntu/dev/ttt/results/ttt_1_bert_base_pt-cased/losses.csv')

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot losses on left y-axis
color1, color2 = 'tab:red', 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='black')
line1 = ax1.plot(df['epoch'], df['train_loss'], color=color1, label='Train Loss', linewidth=2)
line2 = ax1.plot(df['epoch'], df['val_loss'], color=color2, label='Val Loss', linewidth=2)
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)

# Create secondary y-axis for learning rate
ax2 = ax1.twinx()
color3 = 'tab:green'
ax2.set_ylabel('Learning Rate', color=color3)
line3 = ax2.plot(df['epoch'], df['lr'], color=color3, label='Learning Rate', 
                 linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color3)

# Combine legends from both axes
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title('Training Metrics: Loss and Learning Rate')
plt.tight_layout()
plt.savefig('ttt1.png')

# Close the plot to prevent it from being displayed (optional)
plt.close() 