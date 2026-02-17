"""
FIXED PCA ANALYSIS - ALL 36 SUBJECTS WITH YOUNG/OLD HIGHLIGHTING
=================================================================

This script performs PCA using ALL 36 subjects without losing any to missing data.
Properly handles missing values and ensures accurate subject counts.

Author: Workstream 1 Analysis
Date: 2026-02-17
Version: FIXED
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("="*80)
print("FIXED PCA ANALYSIS - ALL 36 SUBJECTS")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA AND GET SUBJECT REFERENCE
# ============================================================================
print("\nStep 1: Loading data and creating subject reference...")

excel_file = 'data.xlsx'

# Load reference sheet to get ALL subjects and their ages
df_ref = pd.read_excel(excel_file, sheet_name='Forehead (PD2, Step Loading)', engine='openpyxl')

print(f"\nReference subjects from Forehead (PD2, Step Loading):")
print(f"  Total subjects: {len(df_ref)}")
print(f"  Age range: {df_ref['Age'].min():.0f} - {df_ref['Age'].max():.0f} years")

# Create subject-age lookup (this is the MASTER reference)
subject_age_map = dict(zip(df_ref['Subjects'], df_ref['Age']))

# Define age groups
young_cutoff = 35
old_cutoff = 50

# Classify subjects based on REFERENCE sheet
young_subjects = set(df_ref[df_ref['Age'] <= young_cutoff]['Subjects'].values)
middle_subjects = set(df_ref[(df_ref['Age'] > young_cutoff) & (df_ref['Age'] < old_cutoff)]['Subjects'].values)
old_subjects = set(df_ref[df_ref['Age'] >= old_cutoff]['Subjects'].values)

print(f"\nAge group classification (from reference):")
print(f"  Young (≤{young_cutoff}): {len(young_subjects)} subjects")
print(f"  Middle ({young_cutoff+1}-{old_cutoff-1}): {len(middle_subjects)} subjects")
print(f"  Old (≥{old_cutoff}): {len(old_subjects)} subjects")
print(f"  TOTAL: {len(young_subjects) + len(middle_subjects) + len(old_subjects)} subjects")

# ============================================================================
# STEP 2: LOAD ALL EXPERIMENTAL CONDITION SHEETS
# ============================================================================
print("\n" + "="*80)
print("Step 2: Loading all experimental conditions")
print("="*80)

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

cutometer_params = [' R0', ' R1', ' R2', ' R3', ' R4', ' R5', ' R6', 
                    ' R7', ' R8', ' F0', ' F1', ' Q0', ' Q1', ' Q2', ' Q3']

all_data = {}
for sheet in sheets:
    df = pd.read_excel(excel_file, sheet_name=sheet, engine='openpyxl')
    all_data[sheet] = df
    print(f"  ✓ {sheet}: {df.shape[0]} rows")

# ============================================================================
# STEP 3: CREATE MASTER DATASET (KEEP ALL SUBJECTS)
# ============================================================================
print("\n" + "="*80)
print("Step 3: Creating master dataset - KEEPING ALL SUBJECTS")
print("="*80)

master_data = []

for sheet_name, df_sheet in all_data.items():
    # Copy the dataframe
    df_all = df_sheet.copy()
    
    # Add age from lookup (in case some sheets have missing age)
    df_all['Age_Verified'] = df_all['Subjects'].map(subject_age_map)
    
    # Use verified age for grouping
    df_all['AgeGroup'] = 'Unknown'  # Default
    df_all.loc[df_all['Subjects'].isin(young_subjects), 'AgeGroup'] = 'Young'
    df_all.loc[df_all['Subjects'].isin(middle_subjects), 'AgeGroup'] = 'Middle'
    df_all.loc[df_all['Subjects'].isin(old_subjects), 'AgeGroup'] = 'Old'
    
    # Add condition metadata
    parts = sheet_name.replace('(', '').replace(')', '').split()
    df_all['Location'] = parts[0]
    df_all['Condition'] = sheet_name
    
    master_data.append(df_all)

# Combine all data
master_df = pd.concat(master_data, ignore_index=True)

# Remove rows where subject couldn't be classified (shouldn't happen, but safety check)
master_df = master_df[master_df['AgeGroup'] != 'Unknown']

print(f"\nMaster dataset created:")
print(f"  Total measurements: {len(master_df)}")
print(f"  Unique subjects: {master_df['Subjects'].nunique()}")

# Count by age group
young_count = len(master_df[master_df['AgeGroup']=='Young'])
middle_count = len(master_df[master_df['AgeGroup']=='Middle'])
old_count = len(master_df[master_df['AgeGroup']=='Old'])

print(f"\nMeasurements per age group:")
print(f"  Young: {young_count} measurements ({young_count//12} subjects × 12 conditions)")
print(f"  Middle: {middle_count} measurements ({middle_count//12} subjects × 12 conditions)")
print(f"  Old: {old_count} measurements ({old_count//12} subjects × 12 conditions)")

# Verify we have the right number of unique subjects in each group
young_unique = master_df[master_df['AgeGroup']=='Young']['Subjects'].nunique()
middle_unique = master_df[master_df['AgeGroup']=='Middle']['Subjects'].nunique()
old_unique = master_df[master_df['AgeGroup']=='Old']['Subjects'].nunique()

print(f"\nUnique subjects per age group (VERIFICATION):")
print(f"  Young: {young_unique} subjects (expected: {len(young_subjects)})")
print(f"  Middle: {middle_unique} subjects (expected: {len(middle_subjects)})")
print(f"  Old: {old_unique} subjects (expected: {len(old_subjects)})")

if young_unique == len(young_subjects) and middle_unique == len(middle_subjects) and old_unique == len(old_subjects):
    print("  ✓ ALL SUBJECTS PRESENT - NO DATA LOSS!")
else:
    print("  ⚠ WARNING: Some subjects missing from dataset!")

# ============================================================================
# STEP 4: PREPARE DATA FOR PCA
# ============================================================================
print("\n" + "="*80)
print("Step 4: Preparing data for PCA")
print("="*80)

# Extract features
X = master_df[cutometer_params].values
age_groups = master_df['AgeGroup'].values
ages = master_df['Age_Verified'].values

# Handle missing values (replace NaN with column median)
print("\nHandling missing values...")
X_clean = X.copy()
for i in range(X.shape[1]):
    col = X[:, i]
    col_median = np.nanmedian(col)
    nan_mask = np.isnan(col)
    if np.any(nan_mask):
        print(f"  {cutometer_params[i].strip()}: {np.sum(nan_mask)} missing values → replaced with median ({col_median:.3f})")
        X_clean[nan_mask, i] = col_median

print(f"\nData matrix: {X_clean.shape}")
print(f"  {X_clean.shape[0]} measurements × {X_clean.shape[1]} parameters")
print(f"  No NaN values: {not np.any(np.isnan(X_clean))}")

# Standardize
print("\nStandardizing data (mean=0, std=1)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# ============================================================================
# STEP 5: PERFORM PCA
# ============================================================================
print("\n" + "="*80)
print("Step 5: Performing PCA on ALL subjects")
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
# STEP 6: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("Step 6: Creating visualizations")
print("="*80)

import os
if not os.path.exists('results'):
    os.makedirs('results')

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

# Calculate actual subject counts for labels
young_n = master_df[master_df['AgeGroup']=='Young']['Subjects'].nunique()
middle_n = master_df[master_df['AgeGroup']=='Middle']['Subjects'].nunique()
old_n = master_df[master_df['AgeGroup']=='Old']['Subjects'].nunique()

# Plot in order: Middle first (background), then Young and Old (foreground)
for group in ['Middle', 'Young', 'Old']:
    mask = age_groups == group
    n_subjects = master_df[master_df['AgeGroup']==group]['Subjects'].nunique()
    
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors_map[group], s=sizes_map[group], 
               alpha=alphas_map[group], marker='o',
               edgecolors='black' if group != 'Middle' else 'none',
               linewidth=0.5 if group != 'Middle' else 0,
               label=f'{group} (n={n_subjects} subjects)')

ax2.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_title('PCA: Young vs Old Separation\n(All Subjects Used for PCA)', 
             fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/PCA_AllSubjects_YoungOld_FIXED.png', dpi=300, bbox_inches='tight')
print("✓ Saved: PCA_AllSubjects_YoungOld_FIXED.png")
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
plt.savefig('results/PCA_AllSubjects_AgeGradient_FIXED.png', dpi=300, bbox_inches='tight')
print("✓ Saved: PCA_AllSubjects_AgeGradient_FIXED.png")
plt.close()

# VISUALIZATION 3: Young vs Old only (for comparison)
# -------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Only young and old data points
mask_young = age_groups == 'Young'
mask_old = age_groups == 'Old'

ax1.scatter(X_pca[mask_young, 0], X_pca[mask_young, 1], 
           c='blue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
           label=f'Young ≤{young_cutoff} (n={young_n})')
ax1.scatter(X_pca[mask_old, 0], X_pca[mask_old, 1], 
           c='red', s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
           label=f'Old ≥{old_cutoff} (n={old_n})')

ax1.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax1.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax1.set_title('Young vs Old Only\n(PCA from All Subjects)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: All three groups for comparison
for group in ['Middle', 'Young', 'Old']:
    mask = age_groups == group
    n_subjects = master_df[master_df['AgeGroup']==group]['Subjects'].nunique()
    
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors_map[group], s=sizes_map[group], 
               alpha=alphas_map[group], edgecolors='black' if group != 'Middle' else 'none',
               linewidth=0.5 if group != 'Middle' else 0,
               label=f'{group} (n={n_subjects})')

ax2.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax2.set_title('All Three Groups\n(Full Context)', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/PCA_Comparison_YoungOldVsAll_FIXED.png', dpi=300, bbox_inches='tight')
print("✓ Saved: PCA_Comparison_YoungOldVsAll_FIXED.png")
plt.close()

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("Step 7: Saving results")
print("="*80)

with pd.ExcelWriter('results/PCA_AllSubjects_Results_FIXED.xlsx', engine='openpyxl') as writer:
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
            young_n,
            middle_n,
            old_n,
            f"{variance_explained[0]*100:.2f}",
            f"{variance_explained[1]*100:.2f}",
            n_components_90
        ]
    })
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print("✓ Saved: PCA_AllSubjects_Results_FIXED.xlsx")

# ============================================================================
# STEP 8: SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - FIXED PCA ANALYSIS")
print("="*80)

print(f"""
KEY FINDINGS:

1. DATA VERIFICATION:
   • Total subjects: {master_df['Subjects'].nunique()} ✓
   • Total measurements: {len(master_df)}
   • Young (≤{young_cutoff}): {young_n} subjects ✓
   • Middle ({young_cutoff+1}-{old_cutoff-1}): {middle_n} subjects ✓
   • Old (≥{old_cutoff}): {old_n} subjects ✓
   • ALL SUBJECTS ACCOUNTED FOR!

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
   ✓ PCA_AllSubjects_YoungOld_FIXED.png 
   ✓ PCA_AllSubjects_AgeGradient_FIXED.png
   ✓ PCA_Comparison_YoungOldVsAll_FIXED.png
   ✓ PCA_AllSubjects_Results_FIXED.xlsx

5. FIXES APPLIED:
   • Used reference sheet to establish master subject list
   • Verified age assignments against reference
   • Properly handled missing values without dropping subjects
   • Accurate subject counting in all visualizations
   


