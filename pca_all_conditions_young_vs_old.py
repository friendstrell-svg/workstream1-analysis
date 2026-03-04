import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

print("="*80)
print("COMPREHENSIVE PCA ANALYSIS: YOUNG VS OLD ACROSS ALL CONDITIONS")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

excel_file = 'data.xlsx'
output_file = 'PCA_Young_vs_Old_ALL_CONDITIONS.xlsx'

# Age thresholds
YOUNG_THRESHOLD = 35
OLD_THRESHOLD = 50

# Parameters
param_cols = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 
              'F0', 'F1', 'Q0', 'Q1', 'Q2', 'Q3']

# All experimental conditions
sheet_names = [
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

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def perform_group_pca(df, group_name, available_params):
    """Perform PCA on a group and return results"""
    
    if len(df) < 3:
        print(f"    ⚠ {group_name}: Only {len(df)} subjects - skipping PCA")
        return None
    
    # Extract data
    X = df[available_params].values
    X_clean = np.nan_to_num(X, nan=np.nanmedian(X))
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Get variance
    var_explained = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_explained)
    
    # Add PC scores to dataframe
    df_with_pcs = df.copy()
    for i in range(min(5, X_pca.shape[1])):
        df_with_pcs[f'PC{i+1}'] = X_pca[:, i]
    
    return {
        'dataframe': df_with_pcs,
        'pca_model': pca,
        'variance': var_explained,
        'cumulative_variance': cum_var,
        'loadings_pc1': pca.components_[0] if len(pca.components_) > 0 else None,
        'n_subjects': len(df)
    }

# ============================================================================
# LOAD AND PROCESS ALL CONDITIONS
# ============================================================================
print("\nStep 1: Processing all experimental conditions...")
print("="*80)

all_results = {}
summary_data = []

for sheet_name in sheet_names:
    print(f"\n📊 Processing: {sheet_name}")
    print("-" * 80)
    
    try:
        # Load sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
        df.columns = df.columns.str.strip()
        
        # Find age column
        age_column = 'Age' if 'Age' in df.columns else None
        if not age_column:
            age_cols = [col for col in df.columns if 'age' in col.lower()]
            if age_cols:
                age_column = age_cols[0]
            else:
                print(f"  ⚠ No age column found - skipping")
                continue
        
        # Find available parameters
        available_params = [col for col in param_cols if col in df.columns]
        if len(available_params) < len(param_cols):
            available_params = [col if col in df.columns else f' {col}' 
                              for col in param_cols 
                              if col in df.columns or f' {col}' in df.columns]
        
        print(f"  Total subjects: {len(df)}")
        print(f"  Parameters available: {len(available_params)}")
        
        # Split into young and old
        df_young = df[df[age_column] <= YOUNG_THRESHOLD].copy()
        df_old = df[df[age_column] >= OLD_THRESHOLD].copy()
        
        print(f"  Young (≤{YOUNG_THRESHOLD}): {len(df_young)} subjects (ages {df_young[age_column].min():.0f}-{df_young[age_column].max():.0f})")
        print(f"  Old (≥{OLD_THRESHOLD}): {len(df_old)} subjects (ages {df_old[age_column].min():.0f}-{df_old[age_column].max():.0f})")
        
        # Perform PCA on each group
        print(f"\n  Running PCA...")
        young_results = perform_group_pca(df_young, "Young", available_params)
        old_results = perform_group_pca(df_old, "Old", available_params)
        
        if young_results and old_results:
            print(f"  ✓ Young PC1: {young_results['variance'][0]:.1f}% | PC1+PC2: {young_results['cumulative_variance'][1]:.1f}%")
            print(f"  ✓ Old PC1: {old_results['variance'][0]:.1f}% | PC1+PC2: {old_results['cumulative_variance'][1]:.1f}%")
            print(f"  ✓ Difference: {abs(young_results['variance'][0] - old_results['variance'][0]):.1f} percentage points")
            
            # Store results
            all_results[sheet_name] = {
                'young': young_results,
                'old': old_results,
                'condition': sheet_name
            }
            
            # Add to summary
            parts = sheet_name.replace('(', '').replace(')', '').replace(',', '').split()
            summary_data.append({
                'Location': parts[0],
                'Probe': parts[1],
                'Loading': ' '.join(parts[2:]),
                'Young_n': young_results['n_subjects'],
                'Old_n': old_results['n_subjects'],
                'Young_PC1_%': young_results['variance'][0],
                'Old_PC1_%': old_results['variance'][0],
                'Diff_PC1_%': young_results['variance'][0] - old_results['variance'][0],
                'Young_PC1+PC2_%': young_results['cumulative_variance'][1],
                'Old_PC1+PC2_%': old_results['cumulative_variance'][1],
                'Young_5PC_%': young_results['cumulative_variance'][4] if len(young_results['cumulative_variance']) > 4 else 100,
                'Old_5PC_%': old_results['cumulative_variance'][4] if len(old_results['cumulative_variance']) > 4 else 100
            })
        else:
            print(f"  ⚠ Insufficient data for PCA comparison")
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}")
        continue

# ============================================================================
# CREATE COMPREHENSIVE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("Step 2: Creating comprehensive summary...")
print("="*80)

summary_df = pd.DataFrame(summary_data)
print(f"\n  Successfully analyzed {len(summary_df)} conditions")

# ============================================================================
# SAVE ALL RESULTS TO EXCEL
# ============================================================================
print("\n" + "="*80)
print("Step 3: Saving results to Excel...")
print("="*80)

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    # Save summary
    summary_df.to_excel(writer, sheet_name='SUMMARY', index=False)
    print("  ✓ Saved: SUMMARY (overview of all conditions)")
    
    # Save each condition's results
    for condition_name, results in all_results.items():
        # Create safe sheet name (Excel limits to 31 chars)
        parts = condition_name.replace('(', '').replace(')', '').replace(',', '').split()
        short_name = f"{parts[0][:3]}_{parts[1]}_{parts[2][:4]}"
        
        # Save young data
        young_sheet = f"{short_name}_Young"
        results['young']['dataframe'].to_excel(writer, sheet_name=young_sheet, index=False)
        print(f"  ✓ Saved: {young_sheet}")
        
        # Save old data
        old_sheet = f"{short_name}_Old"
        results['old']['dataframe'].to_excel(writer, sheet_name=old_sheet, index=False)
        print(f"  ✓ Saved: {old_sheet}")
    
    # Create variance comparison table for all conditions
    variance_comparison_data = []
    for condition_name, results in all_results.items():
        parts = condition_name.replace('(', '').replace(')', '').replace(',', '').split()
        for i in range(5):
            variance_comparison_data.append({
                'Location': parts[0],
                'Probe': parts[1],
                'Loading': ' '.join(parts[2:]),
                'Component': f'PC{i+1}',
                'Young_%': results['young']['variance'][i] if i < len(results['young']['variance']) else 0,
                'Old_%': results['old']['variance'][i] if i < len(results['old']['variance']) else 0,
                'Difference_%': (results['young']['variance'][i] if i < len(results['young']['variance']) else 0) - 
                               (results['old']['variance'][i] if i < len(results['old']['variance']) else 0)
            })
    
    variance_comp_df = pd.DataFrame(variance_comparison_data)
    variance_comp_df.to_excel(writer, sheet_name='Variance_All_Conditions', index=False)
    print("  ✓ Saved: Variance_All_Conditions")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("Step 4: Creating visualizations...")
print("="*80)

# Create comparison plots for each condition
n_conditions = len(all_results)
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.flatten()

for idx, (condition_name, results) in enumerate(all_results.items()):
    if idx >= 12:
        break
    
    ax = axes[idx]
    
    # Bar chart comparing PC1-PC3
    x = np.arange(3)
    width = 0.35
    
    young_var = results['young']['variance'][:3]
    old_var = results['old']['variance'][:3]
    
    ax.bar(x - width/2, young_var, width, label='Young', color='#4CAF50', edgecolor='black')
    ax.bar(x + width/2, old_var, width, label='Old', color='#F44336', edgecolor='black')
    
    # Title with condition info
    parts = condition_name.replace('(', '').replace(')', '').replace(',', '').split()
    title = f"{parts[0]}\n{parts[1]} {parts[2]}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Variance (%)', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['PC1', 'PC2', 'PC3'], fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('PCA Variance Comparison: Young vs Old Across All Conditions', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('PCA_All_Conditions_Comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved: PCA_All_Conditions_Comparison.png")
plt.close()

# Create summary heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Young PC1 variance across conditions
young_pc1_data = summary_df.pivot(index='Location', columns=['Probe', 'Loading'], values='Young_PC1_%')
im1 = axes[0].imshow(young_pc1_data.values, cmap='Greens', aspect='auto')
axes[0].set_title('Young Group: PC1 Variance (%)', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Condition', fontweight='bold')
axes[0].set_ylabel('Location', fontweight='bold')
axes[0].set_xticks(range(len(young_pc1_data.columns)))
axes[0].set_xticklabels([f"{col[0]}\n{col[1]}" for col in young_pc1_data.columns], 
                        rotation=45, ha='right', fontsize=8)
axes[0].set_yticks(range(len(young_pc1_data.index)))
axes[0].set_yticklabels(young_pc1_data.index, fontsize=10)
plt.colorbar(im1, ax=axes[0], label='Variance %')

# Add values
for i in range(len(young_pc1_data.index)):
    for j in range(len(young_pc1_data.columns)):
        text = axes[0].text(j, i, f'{young_pc1_data.values[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

# Plot 2: Old PC1 variance
old_pc1_data = summary_df.pivot(index='Location', columns=['Probe', 'Loading'], values='Old_PC1_%')
im2 = axes[1].imshow(old_pc1_data.values, cmap='Reds', aspect='auto')
axes[1].set_title('Old Group: PC1 Variance (%)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Condition', fontweight='bold')
axes[1].set_ylabel('Location', fontweight='bold')
axes[1].set_xticks(range(len(old_pc1_data.columns)))
axes[1].set_xticklabels([f"{col[0]}\n{col[1]}" for col in old_pc1_data.columns], 
                       rotation=45, ha='right', fontsize=8)
axes[1].set_yticks(range(len(old_pc1_data.index)))
axes[1].set_yticklabels(old_pc1_data.index, fontsize=10)
plt.colorbar(im2, ax=axes[1], label='Variance %')

for i in range(len(old_pc1_data.index)):
    for j in range(len(old_pc1_data.columns)):
        text = axes[1].text(j, i, f'{old_pc1_data.values[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

# Plot 3: Difference
diff_data = summary_df.pivot(index='Location', columns=['Probe', 'Loading'], values='Diff_PC1_%')
im3 = axes[2].imshow(diff_data.values, cmap='RdBu_r', aspect='auto', vmin=-10, vmax=10)
axes[2].set_title('Difference: Young - Old (%)', fontweight='bold', fontsize=12)
axes[2].set_xlabel('Condition', fontweight='bold')
axes[2].set_ylabel('Location', fontweight='bold')
axes[2].set_xticks(range(len(diff_data.columns)))
axes[2].set_xticklabels([f"{col[0]}\n{col[1]}" for col in diff_data.columns], 
                       rotation=45, ha='right', fontsize=8)
axes[2].set_yticks(range(len(diff_data.index)))
axes[2].set_yticklabels(diff_data.index, fontsize=10)
plt.colorbar(im3, ax=axes[2], label='Difference %')

for i in range(len(diff_data.index)):
    for j in range(len(diff_data.columns)):
        text = axes[2].text(j, i, f'{diff_data.values[i, j]:+.1f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('PCA_Heatmap_Summary.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved: PCA_Heatmap_Summary.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)

print(f"""
ANALYSIS COMPLETE FOR {len(all_results)} CONDITIONS!

OUTPUT FILES:

1. {output_file}
   • SUMMARY sheet - Overview of all conditions
   • {len(all_results)*2} data sheets - Young and Old for each condition
   • Variance_All_Conditions - Complete variance breakdown

2. PCA_All_Conditions_Comparison.png
   • 12 subplots showing Young vs Old for each condition

3. PCA_Heatmap_Summary.png
   • Heatmaps showing PC1 variance across all conditions
   • Young, Old, and Difference heatmaps

KEY FINDINGS:

Average PC1 variance:
  Young: {summary_df['Young_PC1_%'].mean():.1f}% (range: {summary_df['Young_PC1_%'].min():.1f}-{summary_df['Young_PC1_%'].max():.1f}%)
  Old:   {summary_df['Old_PC1_%'].mean():.1f}% (range: {summary_df['Old_PC1_%'].min():.1f}-{summary_df['Old_PC1_%'].max():.1f}%)

Largest difference:
  {summary_df.loc[summary_df['Diff_PC1_%'].abs().idxmax(), 'Location']} - {summary_df.loc[summary_df['Diff_PC1_%'].abs().idxmax(), 'Probe']} - {summary_df.loc[summary_df['Diff_PC1_%'].abs().idxmax(), 'Loading']}
  Difference: {summary_df['Diff_PC1_%'].abs().max():.1f} percentage points
""")

print("="*80)
print("All analyses complete!")
print("="*80)
