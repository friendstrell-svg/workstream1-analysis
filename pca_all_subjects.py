"""
IMPROVED PCA ANALYSIS - ALL SUBJECTS WITH YOUNG/OLD HIGHLIGHTING
================================================================

This script performs PCA using ALL 36 subjects but highlights young vs old
groups in the visualization. This is more statistically rigorous than
performing PCA only on age extremes.

Author: Workstream 1 Analysis
Date: 2026-02-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("="*80)
print("IMPROVED PCA ANALYSIS - ALL SUBJECTS")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nStep 1: Loading data...")

excel_file = 'data.xlsx'

# Define the 12 experimental condition sheets
sheets = [
    'Forehead (PD2, Step Loading)',
    'Forehead (PD2, Ramp Loading)',
    'Forehead (PD8, Step Loading)',
    'Forehead (PD8, Ramp Loading)',
    'Parotid (PD2, Step Loading)',
    'Parotid (PD2, Ramp Loading)',
    'Parotid (PD8, Step Loading)',
    'Parotid (PD8, Ramp Loading)',
    'Jaw (PD2, Step Loading)',
    'Jaw (PD2, Ramp Loading)',
    'Jaw (PD8, Step Loading)',
    'Jaw (PD8, Ramp Loading)'
]

# Load all sheets
all_data = {}
for sheet in sheets:
    df = pd.read_excel(excel_file, sheet_name=sheet, engine='openpyxl')
    all_data[sheet] = df
    print(f"  ✓ {sheet}: {df.shape}")

# ============================================================================
# STEP 2: CREATE MASTER DATASET WITH ALL SUBJECTS
# ============================================================================
print("\n" + "="*80)
print("Step 2: Creating master dataset with ALL subjects")
print("="*80)

# Cutometer parameters
cutometer_params = [' R0', ' R1', ' R2', ' R3', ' R4', ' R5', ' R6', 
                    ' R7', ' R8', ' F0', ' F1', ' Q0', ' Q1', ' Q2', ' Q3']

# Get age reference
df_ref = all_data['Forehead (PD2, Step Loading)']

print(f"\nTotal subjects: {len(df_ref)}")
print(f"Age range: {df_ref['Age'].min():.0f} - {df_ref['Age'].max():.0f} years")

# Define age groups for visualization
young_cutoff = 35
old_cutoff = 50

young_subjects = df_ref[df_ref['Age'] <= young_cutoff]['Subjects'].values
middle_subjects = df_ref[(df_ref['Age'] > young_cutoff) & (df_ref['Age'] < old_cutoff)]['Subjects'].values
old_subjects = df_ref[df_ref['Age'] >= old_cutoff]['Subjects'].values

print(f"\nAge groups:")
print(f"  Young (≤{young_cutoff}): {len(young_subjects)} subjects")
print(f"  Middle ({young_cutoff+1}-{old_cutoff-1}): {len(middle_subjects)} subjects")
print(f"  Old (≥{old_cutoff}): {len(old_subjects)} subjects")

# Create master dataset with ALL subjects (no filtering)
master_data = []

for sheet_name, df_sheet in all_data.items():
    # Take ALL subjects - no exclusions
    df_all = df_sheet.copy()
    
    # Add age group labels for visualization
    df_all['AgeGroup'] = 'Middle'  # Default
    df_all.loc[df_all['Subjects'].isin(young_subjects), 'AgeGroup'] = 'Young'
    df_all.loc[df_all['Subjects'].isin(old_subjects), 'AgeGroup'] = 'Old'
    
    master_data.append(df_all)

master_df = pd.concat(master_data, ignore_index=True)

print(f"\nMaster dataset created:")
print(f"  Total measurements: {len(master_df)}")
print(f"  Young: {len(master_df[master_df['AgeGroup']=='Young'])}")
print(f"  Middle: {len(master_df[master_df['AgeGroup']=='Middle'])}")
print(f"  Old: {len(master_df[master_df['AgeGroup']=='Old'])}")

# ============================================================================
# STEP 3: PREPARE DATA FOR PCA
# ============================================================================
print("\n" + "="*80)
print("Step 3: Preparing data for PCA")
print("="*80)

# Extract features
X = master_df[cutometer_params].values
age_groups = master_df['AgeGroup'].values
ages = master_df['Age'].values

# Handle missing values
X_clean = np.nan_to_num(X, nan=np.nanmedian(X))

print(f"\nData matrix: {X_clean.shape}")
print(f"  {X_clean.shape[0]} measurements × {X_clean.shape[1]} parameters")

# Standardize
print("\nStandardizing data (mean=0, std=1)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# ============================================================================
# STEP 4: PERFORM PCA
# ============================================================================
print("\n" + "="*80)
print("Step 4: Performing PCA on ALL subjects")
print("="*80)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

variance_explained = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print("\nVariance explained by each PC:")
for i in range(min(8, len(variance_explained))):
    print(f"  PC{i+1}: {variance_explained[i]*100:6.2f}% (Cumulative: {cumulative_variance[i]*100:6.2f}%)")

n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"\n✓ {n_components_90} components explain 90% of variance")

# Get PC loadings
param_names = [p.strip() for p in cutometer_params]
pc1_loadings = pca.components_[0]
pc1_loading_df = pd.DataFrame({
    'Parameter': param_names,
    'PC1_Loading': np.abs(pc1_loadings)
}).sort_values('PC1_Loading', ascending=False)

print("\nTop 5 contributors to PC1:")
for _, row in pc1_loading_df.head(5).iterrows():
    print(f"  {row['Parameter']}: {row['PC1_Loading']:.3f}")

# ============================================================================
# STEP 5: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("Step 5: Creating visualizations")
print("="*80)

# VISUALIZATION 1: Classic PCA plot with Young/Old/Middle
# -------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Scree plot
ax1.bar(range(1, len(variance_explained)+1), variance_explained*100, 
        color='steelblue', alpha=0.8, edgecolor='black')
ax1.plot(range(1, len(variance_explained)+1), cumulative_variance*100, 
         'ro-', linewidth=2, markersize=8, label='Cumulative')
ax1.axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% threshold')
ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
ax1.set_title('PCA Scree Plot (All 36 Subjects)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(range(1, 16))

# Right: PC1 vs PC2 scatter with three groups
colors_map = {'Young': 'blue', 'Middle': 'gray', 'Old': 'red'}
sizes_map = {'Young': 50, 'Middle': 20, 'Old': 50}
alphas_map = {'Young': 0.7, 'Middle': 0.3, 'Old': 0.7}
markers_map = {'Young': 'o', 'Middle': 'o', 'Old': 'o'}

# Plot in order: Middle first (background), then Young and Old (foreground)
for group in ['Middle', 'Young', 'Old']:
    mask = age_groups == group
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors_map[group], s=sizes_map[group], 
               alpha=alphas_map[group], marker=markers_map[group],
               edgecolors='black' if group != 'Middle' else 'none',
               linewidth=0.5 if group != 'Middle' else 0,
               label=f'{group} (n={np.sum(mask)//12:.0f} subjects)')

ax2.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_title('PCA: Young vs Old Separation\n(All Subjects Used for PCA)', 
             fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/PCA_AllSubjects_YoungOld.png', dpi=300, bbox_inches='tight')
print("✓ Saved: PCA_AllSubjects_YoungOld.png")
plt.close()

# VISUALIZATION 2: Continuous age gradient
# ----------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                    c=ages, cmap='RdYlBu_r', s=50, 
                    alpha=0.6, edgecolors='black', linewidth=0.5)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Age (years)', fontsize=12, fontweight='bold')

ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax.set_title('PCA: Continuous Age Gradient\n(All 36 Subjects)', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/PCA_AllSubjects_AgeGradient.png', dpi=300, bbox_inches='tight')
print("✓ Saved: PCA_AllSubjects_AgeGradient.png")
plt.close()

# VISUALIZATION 3: Young vs Old only (for comparison with old method)
# -------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Only young and old data points
mask_young = age_groups == 'Young'
mask_old = age_groups == 'Old'

ax1.scatter(X_pca[mask_young, 0], X_pca[mask_young, 1], 
           c='blue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
           label=f'Young ≤{young_cutoff} (n={np.sum(mask_young)//12:.0f})')
ax1.scatter(X_pca[mask_old, 0], X_pca[mask_old, 1], 
           c='red', s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
           label=f'Old ≥{old_cutoff} (n={np.sum(mask_old)//12:.0f})')

ax1.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax1.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax1.set_title('Young vs Old Only\n(PCA from All Subjects)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: All three groups for comparison
for group in ['Middle', 'Young', 'Old']:
    mask = age_groups == group
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors_map[group], s=sizes_map[group], 
               alpha=alphas_map[group], edgecolors='black' if group != 'Middle' else 'none',
               linewidth=0.5 if group != 'Middle' else 0,
               label=f'{group} (n={np.sum(mask)//12:.0f})')

ax2.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_title('All Three Groups\n(Full Context)', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/PCA_Comparison_YoungOldVsAll.png', dpi=300, bbox_inches='tight')
print("✓ Saved: PCA_Comparison_YoungOldVsAll.png")
plt.close()

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("Step 6: Saving results")
print("="*80)

# Save PCA results to Excel
import os
if not os.path.exists('results'):
    os.makedirs('results')

with pd.ExcelWriter('results/PCA_AllSubjects_Results.xlsx', engine='openpyxl') as writer:
    # Variance explained
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(variance_explained))],
        'Variance_Explained_%': variance_explained * 100,
        'Cumulative_%': cumulative_variance * 100
    })
    variance_df.to_excel(writer, sheet_name='Variance_Explained', index=False)
    
    # PC1 loadings
    pc1_full_loadings = pd.DataFrame({
        'Parameter': param_names,
        'PC1_Loading': pca.components_[0],
        'Abs_Loading': np.abs(pca.components_[0])
    }).sort_values('Abs_Loading', ascending=False)
    pc1_full_loadings.to_excel(writer, sheet_name='PC1_Loadings', index=False)
    
    # PC2 loadings
    pc2_full_loadings = pd.DataFrame({
        'Parameter': param_names,
        'PC2_Loading': pca.components_[1],
        'Abs_Loading': np.abs(pca.components_[1])
    }).sort_values('Abs_Loading', ascending=False)
    pc2_full_loadings.to_excel(writer, sheet_name='PC2_Loadings', index=False)
    
    # Summary
    summary_df = pd.DataFrame({
        'Metric': [
            'Total Subjects',
            'Total Measurements',
            'Young Subjects',
            'Middle Subjects',
            'Old Subjects',
            'PC1 Variance %',
            'PC2 Variance %',
            'PCs for 90% Variance'
        ],
        'Value': [
            master_df['Subjects'].nunique(),
            len(master_df),
            len(young_subjects),
            len(middle_subjects),
            len(old_subjects),
            f"{variance_explained[0]*100:.2f}",
            f"{variance_explained[1]*100:.2f}",
            n_components_90
        ]
    })
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print("✓ Saved: PCA_AllSubjects_Results.xlsx")

# ============================================================================
# STEP 7: SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - IMPROVED PCA ANALYSIS")
print("="*80)

print(f"""
KEY FINDINGS:

1. DATA USED:
   • Total subjects: {master_df['Subjects'].nunique()} (ALL subjects, no exclusions)
   • Total measurements: {len(master_df)}
   • Young (≤{young_cutoff}): {len(young_subjects)} subjects
   • Middle ({young_cutoff+1}-{old_cutoff-1}): {len(middle_subjects)} subjects
   • Old (≥{old_cutoff}): {len(old_subjects)} subjects

2. PCA RESULTS:
   • PC1 explains {variance_explained[0]*100:.1f}% of variance
   • PC2 explains {variance_explained[1]*100:.1f}% of variance
   • {n_components_90} PCs needed for 90% variance

3. TOP CONTRIBUTORS TO PC1:
""")

for i, (_, row) in enumerate(pc1_loading_df.head(5).iterrows(), 1):
    print(f"   {i}. {row['Parameter']}: {row['PC1_Loading']:.3f}")

print(f"""
4. FILES CREATED:
   ✓ PCA_AllSubjects_YoungOld.png (Young/Old highlighted, middle in gray)
   ✓ PCA_AllSubjects_AgeGradient.png (Continuous age color gradient)
   ✓ PCA_Comparison_YoungOldVsAll.png (Side-by-side comparison)
   ✓ PCA_AllSubjects_Results.xlsx (Detailed results)

5. ADVANTAGES OF THIS APPROACH:
   • More robust: PCA based on {master_df['Subjects'].nunique()} subjects instead of 20
   • More honest: Uses all data, doesn't hide middle-age subjects
   • Better context: Shows where young/old sit in full population
   • More generalizable: PCs represent complete variation

RECOMMENDATION FOR REPORT:
Use "PCA_AllSubjects_YoungOld.png" as your main PCA figure.
This shows young vs old separation while being honest about using all data.
""")

print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
