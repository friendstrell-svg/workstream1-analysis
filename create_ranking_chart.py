"""
Create Parameter Ranking Chart for WS1 Poster
Shows all 15 parameters with top 7 highlighted in green
"""

import matplotlib.pyplot as plt
import numpy as np

# Data - your actual combined scores
parameters = ['R2', 'R3', 'R1', 'F1', 'R4', 'R7', 'R6', 'R0', 'Q0', 'R8', 'Q1', 'Q3', 'F0', 'Q2', 'R5']
scores = [0.969, 0.890, 0.844, 0.837, 0.793, 0.616, 0.613, 0.419, 0.420, 0.400, 0.386, 0.385, 0.327, 0.400, 0.242]
selected = [True, True, True, True, True, True, True, False, False, False, False, False, False, False, False]

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Colors - green for selected, gray for excluded
colors = ['#2E7D32' if sel else '#BDBDBD' for sel in selected]

# Create horizontal bar chart
y_pos = np.arange(len(parameters))
bars = ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# Add labels
ax.set_yticks(y_pos)
ax.set_yticklabels(parameters, fontsize=14, fontweight='bold')
ax.set_xlabel('Combined Importance Score', fontsize=14, fontweight='bold')
ax.set_title('Parameter Ranking: 15 → 7 (53% Reduction)', fontsize=16, fontweight='bold', pad=15)

# Add score values at end of bars
for i, (score, sel) in enumerate(zip(scores, selected)):
    label = f'{score:.3f} ⭐' if sel else f'{score:.3f}'
    color = 'black' if sel else 'gray'
    ax.text(score + 0.02, i, label, va='center', fontsize=11, 
            fontweight='bold' if sel else 'normal', color=color)

# Add vertical line at selection threshold
ax.axvline(x=0.60, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Selection threshold')

# Grid
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, 1.1)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E7D32', alpha=0.85, edgecolor='black', label='Selected (n=7)'),
    Patch(facecolor='#BDBDBD', alpha=0.85, edgecolor='black', label='Excluded (n=8)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95)

# Tight layout
plt.tight_layout()

# Save to results folder
import os
if not os.path.exists('results'):
    os.makedirs('results')
    
plt.savefig('results/WS1_Parameter_Ranking_Full.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: results/WS1_Parameter_Ranking_Full.png")

# Show plot
plt.show()