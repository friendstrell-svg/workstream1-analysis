import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

print("="*80)
print("SEPARATE PCA ANALYSIS: YOUNG VS OLD")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

excel_file = 'data.xlsx'
output_file = 'PCA_Young_vs_Old_Separate.xlsx'

# Age thresholds
YOUNG_THRESHOLD = 35
OLD_THRESHOLD = 50

# Parameters
param_cols = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 
              'F0', 'F1', 'Q0', 'Q1', 'Q2', 'Q3']

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nStep 1: Loading data...")

xl = pd.ExcelFile(excel_file, engine='openpyxl')
sheet_names = xl.sheet_names

print(f"  Found {len(sheet_names)} sheets")

# Use first sheet for demonstration (or combine all sheets)
# For this analysis, let's use Forehead PD2 Step Loading as primary
primary_sheet = 'Forehead (PD2, Step Loading)'

df = pd.read_excel(excel_file, sheet_name=primary_sheet, engine='openpyxl')
df.columns = df.columns.str.strip()

print(f"  Loaded sheet: {primary_sheet}")
print(f"  Total subjects: {len(df)}")

# ============================================================================
# SPLIT INTO YOUNG AND OLD GROUPS
# ============================================================================
print("\nStep 2: Splitting into Young vs Old groups...")

# Find age column
if 'Age' not in df.columns:
    print("  ⚠ Warning: No Age column found. Checking alternatives...")
    age_col = [col for col in df.columns if 'age' in col.lower()]
    if age_col:
        age_column = age_col[0]
    else:
        print("  ❌ Cannot find age information!")
        exit(1)
else:
    age_column = 'Age'

# Split groups
df_young = df[df[age_column] <= YOUNG_THRESHOLD].copy()
df_old = df[df[age_column] >= OLD_THRESHOLD].copy()

print(f"  Young group (≤{YOUNG_THRESHOLD}): {len(df_young)} subjects")
print(f"    Age range: {df_young[age_column].min():.0f} - {df_young[age_column].max():.0f}")
print(f"  Old group (≥{OLD_THRESHOLD}): {len(df_old)} subjects")
print(f"    Age range: {df_old[age_column].min():.0f} - {df_old[age_column].max():.0f}")

# ============================================================================
# FIND AVAILABLE PARAMETERS
# ============================================================================

# Find which parameters exist
available_params = [col for col in param_cols if col in df.columns]
if len(available_params) < len(param_cols):
    available_params = [col if col in df.columns else f' {col}' 
                       for col in param_cols 
                       if col in df.columns or f' {col}' in df.columns]

print(f"\n  Using {len(available_params)} parameters for PCA")

# ============================================================================
# PCA ON YOUNG GROUP
# ============================================================================
print("\n" + "="*80)
print("Step 3: PCA on YOUNG group")
print("="*80)

# Extract young data
X_young = df_young[available_params].values
X_young_clean = np.nan_to_num(X_young, nan=np.nanmedian(X_young))

# Standardize
scaler_young = StandardScaler()
X_young_scaled = scaler_young.fit_transform(X_young_clean)

# PCA
pca_young = PCA()
X_young_pca = pca_young.fit_transform(X_young_scaled)

# Variance explained
var_young = pca_young.explained_variance_ratio_ * 100
cum_var_young = np.cumsum(var_young)

print("\n  Variance explained by YOUNG group:")
print("  " + "-"*60)
for i in range(min(5, len(var_young))):
    print(f"  PC{i+1}: {var_young[i]:>6.2f}%  (Cumulative: {cum_var_young[i]:>6.2f}%)")
print("  " + "-"*60)

# Add PC scores to young dataframe
for i in range(min(5, X_young_pca.shape[1])):
    df_young[f'PC{i+1}'] = X_young_pca[:, i]

# ============================================================================
# PCA ON OLD GROUP
# ============================================================================
print("\n" + "="*80)
print("Step 4: PCA on OLD group")
print("="*80)

# Extract old data
X_old = df_old[available_params].values
X_old_clean = np.nan_to_num(X_old, nan=np.nanmedian(X_old))

# Standardize
scaler_old = StandardScaler()
X_old_scaled = scaler_old.fit_transform(X_old_clean)

# PCA
pca_old = PCA()
X_old_pca = pca_old.fit_transform(X_old_scaled)

# Variance explained
var_old = pca_old.explained_variance_ratio_ * 100
cum_var_old = np.cumsum(var_old)

print("\n  Variance explained by OLD group:")
print("  " + "-"*60)
for i in range(min(5, len(var_old))):
    print(f"  PC{i+1}: {var_old[i]:>6.2f}%  (Cumulative: {cum_var_old[i]:>6.2f}%)")
print("  " + "-"*60)

# Add PC scores to old dataframe
for i in range(min(5, X_old_pca.shape[1])):
    df_old[f'PC{i+1}'] = X_old_pca[:, i]

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("Step 5: Comparing Young vs Old PCA structures")
print("="*80)

comparison_data = []
for i in range(min(5, len(var_young), len(var_old))):
    comparison_data.append({
        'Component': f'PC{i+1}',
        'Young_Variance_%': var_young[i],
        'Old_Variance_%': var_old[i],
        'Difference_%': var_young[i] - var_old[i],
        'Young_Cumulative_%': cum_var_young[i],
        'Old_Cumulative_%': cum_var_old[i]
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# LOADINGS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("Step 6: Comparing PC1 loadings between groups")
print("="*80)

# Get PC1 loadings for both groups
loadings_young_pc1 = pca_young.components_[0]
loadings_old_pc1 = pca_old.components_[0]

loadings_comparison = pd.DataFrame({
    'Parameter': [p.strip() for p in available_params],
    'Young_PC1_Loading': loadings_young_pc1,
    'Old_PC1_Loading': loadings_old_pc1,
    'Difference': loadings_young_pc1 - loadings_old_pc1,
    'Abs_Difference': np.abs(loadings_young_pc1 - loadings_old_pc1)
})
loadings_comparison = loadings_comparison.sort_values('Abs_Difference', ascending=False)

print("\n  Top 5 parameters with biggest loading differences:")
print(loadings_comparison.head().to_string(index=False))

# ============================================================================
# SAVE RESULTS TO EXCEL
# ============================================================================
print("\n" + "="*80)
print("Step 7: Saving results to Excel")
print("="*80)

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    # Save young group data with PC scores
    df_young.to_excel(writer, sheet_name='Young_Group_Data', index=False)
    print("  ✓ Saved: Young_Group_Data")
    
    # Save old group data with PC scores
    df_old.to_excel(writer, sheet_name='Old_Group_Data', index=False)
    print("  ✓ Saved: Old_Group_Data")
    
    # Save variance comparison
    comparison_df.to_excel(writer, sheet_name='Variance_Comparison', index=False)
    print("  ✓ Saved: Variance_Comparison")
    
    # Save loadings comparison
    loadings_comparison.to_excel(writer, sheet_name='Loadings_Comparison', index=False)
    print("  ✓ Saved: Loadings_Comparison")
    
    # Save young variance details
    var_details_young = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(var_young))],
        'Variance_%': var_young,
        'Cumulative_%': cum_var_young
    })
    var_details_young.to_excel(writer, sheet_name='Young_Variance_Detail', index=False)
    print("  ✓ Saved: Young_Variance_Detail")
    
    # Save old variance details
    var_details_old = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(var_old))],
        'Variance_%': var_old,
        'Cumulative_%': cum_var_old
    })
    var_details_old.to_excel(writer, sheet_name='Old_Variance_Detail', index=False)
    print("  ✓ Saved: Old_Variance_Detail")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("Step 8: Creating comparison visualizations")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Variance comparison
ax1 = axes[0, 0]
x = np.arange(min(5, len(var_young)))
width = 0.35
ax1.bar(x - width/2, var_young[:5], width, label='Young', color='#4CAF50', edgecolor='black')
ax1.bar(x + width/2, var_old[:5], width, label='Old', color='#F44336', edgecolor='black')
ax1.set_xlabel('Principal Component', fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontweight='bold')
ax1.set_title('Variance Comparison: Young vs Old', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([f'PC{i+1}' for i in range(5)])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Cumulative variance
ax2 = axes[0, 1]
ax2.plot(range(1, 6), cum_var_young[:5], marker='o', linewidth=2, 
         markersize=8, label='Young', color='#4CAF50')
ax2.plot(range(1, 6), cum_var_old[:5], marker='s', linewidth=2, 
         markersize=8, label='Old', color='#F44336')
ax2.set_xlabel('Number of Components', fontweight='bold')
ax2.set_ylabel('Cumulative Variance (%)', fontweight='bold')
ax2.set_title('Cumulative Variance: Young vs Old', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xticks(range(1, 6))

# Plot 3: PC1 scatter (Young)
ax3 = axes[1, 0]
scatter_young = ax3.scatter(df_young['PC1'], df_young['PC2'], 
                           c=df_young[age_column], cmap='Greens', 
                           s=100, edgecolors='black', linewidth=1.5)
ax3.set_xlabel('PC1', fontweight='bold')
ax3.set_ylabel('PC2', fontweight='bold')
ax3.set_title(f'Young Group PCA (n={len(df_young)})', fontweight='bold', fontsize=12)
ax3.grid(alpha=0.3)
plt.colorbar(scatter_young, ax=ax3, label='Age')

# Plot 4: PC1 scatter (Old)
ax4 = axes[1, 1]
scatter_old = ax4.scatter(df_old['PC1'], df_old['PC2'], 
                         c=df_old[age_column], cmap='Reds', 
                         s=100, edgecolors='black', linewidth=1.5)
ax4.set_xlabel('PC1', fontweight='bold')
ax4.set_ylabel('PC2', fontweight='bold')
ax4.set_title(f'Old Group PCA (n={len(df_old)})', fontweight='bold', fontsize=12)
ax4.grid(alpha=0.3)
plt.colorbar(scatter_old, ax=ax4, label='Age')

plt.tight_layout()
plt.savefig('PCA_Young_vs_Old_Comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved: PCA_Young_vs_Old_Comparison.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)

print(f"""
OUTPUT FILES CREATED:

1. {output_file}
   Sheets:
   • Young_Group_Data - Young subjects with PC1-PC5 scores
   • Old_Group_Data - Old subjects with PC1-PC5 scores
   • Variance_Comparison - Side-by-side variance comparison
   • Loadings_Comparison - PC1 loading differences
   • Young_Variance_Detail - Full variance breakdown (Young)
   • Old_Variance_Detail - Full variance breakdown (Old)

2. PCA_Young_vs_Old_Comparison.png
   • Variance comparison charts
   • Cumulative variance curves
   • PCA scatter plots for both groups

KEY FINDINGS:

Young Group ({len(df_young)} subjects):
  • PC1 explains {var_young[0]:.1f}% variance
  • PC1+PC2 explain {cum_var_young[1]:.1f}% variance
  • PC1-PC5 explain {cum_var_young[4]:.1f}% variance

Old Group ({len(df_old)} subjects):
  • PC1 explains {var_old[0]:.1f}% variance
  • PC1+PC2 explain {cum_var_old[1]:.1f}% variance
  • PC1-PC5 explain {cum_var_old[4]:.1f}% variance

Difference:
  • PC1 variance differs by {abs(var_young[0] - var_old[0]):.1f} percentage points
  • {'Young' if var_young[0] > var_old[0] else 'Old'} group has higher PC1 variance
""")

print("="*80)
print("Analysis complete!")
print("="*80)
