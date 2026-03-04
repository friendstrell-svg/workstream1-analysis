
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os




excel_file = 'data.xlsx'
output_file = 'data_with_ALL_PCs.xlsx'

# Parameters to use for PCA
param_cols = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 
              'F0', 'F1', 'Q0', 'Q1', 'Q2', 'Q3']

# How many PCs to save to Excel (saves first 5 for 90% variance)
n_pcs_to_save = 5


print("\nStep 1: Loading data...")

xl = pd.ExcelFile(excel_file, engine='openpyxl')
sheet_names = xl.sheet_names

print(f"  Found {len(sheet_names)} sheets")


print("\nStep 2: Analysing variance structure...")

# Use first sheet to analyse overall variance pattern
first_sheet = sheet_names[0]
df_sample = pd.read_excel(excel_file, sheet_name=first_sheet, engine='openpyxl')
df_sample.columns = df_sample.columns.str.strip()

# Find available parameters
available_params = [col for col in param_cols if col in df_sample.columns]
if len(available_params) < len(param_cols):
    available_params = [col if col in df_sample.columns else f' {col}' 
                       for col in param_cols 
                       if col in df_sample.columns or f' {col}' in df_sample.columns]

print(f"  Using {len(available_params)} parameters for PCA")

# Extract and prepare data
X_sample = df_sample[available_params].values
X_sample_clean = np.nan_to_num(X_sample, nan=np.nanmedian(X_sample))
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample_clean)

# Perform FULL PCA (all components)
pca_full = PCA()
pca_full.fit(X_sample_scaled)

# Get variance explained by each component
variance_explained = pca_full.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(variance_explained)

print("\n" + "-"*80)
print("VARIANCE EXPLAINED BY PRINCIPAL COMPONENTS")
print("-"*80)
print(f"{'PC':<6} {'Variance %':<12} {'Cumulative %':<15} {'Bar Chart'}")
print("-"*80)

for i in range(len(variance_explained)):
    bar_length = int(variance_explained[i] / 2)  # Scale for display
    bar = '█' * bar_length
    
    # Highlight PC1 and PC2 (for visualization)
    marker = " ← VISUALISATION" if i < 2 else ""
    marker = " ← 90% THRESHOLD" if i == 4 else marker
    
    print(f"PC{i+1:<4} {variance_explained[i]:>6.1f}%     {cumulative_variance[i]:>6.1f}%        {bar}{marker}")
    
    if i == 4:  # After PC5
        print("-"*80)

print("-"*80)
print(f"\nKEY FINDINGS:")
print(f"  • PC1 + PC2 (for visualisation): {cumulative_variance[1]:.1f}% variance")
print(f"  • PC1 to PC5 (recommended): {cumulative_variance[4]:.1f}% variance")
print(f"  • All {len(variance_explained)} PCs: 100.0% variance")


print("\n" + "="*80)
print("Step 3: Processing each sheet with PCA...")
print("="*80)

# Create Excel writer
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    variance_summary_data = []
    
    for sheet_name in sheet_names:
        print(f"\n  Processing: {sheet_name}")
        
        # Load sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
        df.columns = df.columns.str.strip()
        
        # Find available parameters
        available_params = [col for col in param_cols if col in df.columns]
        if len(available_params) < len(param_cols):
            available_params = [col if col in df.columns else f' {col}' 
                              for col in param_cols 
                              if col in df.columns or f' {col}' in df.columns]
        
        if len(available_params) < 10:
            print(f"    ⚠ Only {len(available_params)} parameters - skipping")
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            continue
        
        try:
            # Extract and prepare data
            X = df[available_params].values
            X_clean = np.nan_to_num(X, nan=np.nanmedian(X))
            
            # Standardise
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Perform FULL PCA
            pca = PCA()
            X_pca_all = pca.fit_transform(X_scaled)
            
            # Add first 5 PCs to dataframe (for better variance capture)
            for i in range(min(n_pcs_to_save, X_pca_all.shape[1])):
                df[f'PC{i+1}'] = X_pca_all[:, i]
            
            # Store variance info
            var_exp = pca.explained_variance_ratio_ * 100
            cum_var = np.cumsum(var_exp)
            
            variance_summary_data.append({
                'Condition': sheet_name,
                'PC1_%': var_exp[0],
                'PC2_%': var_exp[1],
                'PC1+PC2_%': cum_var[1],
                'PC1-PC5_%': cum_var[4] if len(cum_var) > 4 else cum_var[-1],
                'n_components': len(var_exp)
            })
            
            print(f"    ✓ PC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}% | PC1+PC2: {cum_var[1]:.1f}%")
            print(f"    ✓ PC1-PC5: {cum_var[4]:.1f}% variance")
            print(f"    ✓ Saved PC1-PC{n_pcs_to_save} to Excel")
            
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:80]}")
        
        # Save sheet
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Create variance summary sheet
    if variance_summary_data:
        var_summary_df = pd.DataFrame(variance_summary_data)
        var_summary_df.to_excel(writer, sheet_name='PCA_Variance_Summary', index=False)
        print(f"\n  ✓ Created PCA_Variance_Summary sheet")


print("\n" + "="*80)
print("Step 4: Creating visualisation guide...")
print("="*80)

# Create a simple variance chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Individual variance
ax1.bar(range(1, len(variance_explained)+1), variance_explained, 
        color=['#2E7D32' if i < 2 else '#1976D2' if i < 5 else '#90CAF9' 
               for i in range(len(variance_explained))],
        edgecolor='black', linewidth=1.5)
ax1.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
ax1.set_title('Individual Variance by Component', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(range(1, len(variance_explained)+1))

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E7D32', label='PC1-PC2 (Visualisation)'),
    Patch(facecolor='#1976D2', label='PC3-PC5 (90% variance)'),
    Patch(facecolor='#90CAF9', label='PC6+ (Minimal gain)')
]
ax1.legend(handles=legend_elements, loc='upper right')

# Right plot: Cumulative variance
ax2.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 
         marker='o', linewidth=3, markersize=8, color='#1976D2')
ax2.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% threshold')
ax2.axhline(y=cumulative_variance[1], color='green', linestyle='--', 
            linewidth=2, label=f'PC1+PC2 ({cumulative_variance[1]:.1f}%)')
ax2.fill_between(range(1, len(cumulative_variance)+1), 0, cumulative_variance, 
                  alpha=0.3, color='#1976D2')
ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.legend()
ax2.set_xticks(range(1, len(cumulative_variance)+1))
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('PCA_Variance_Analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved: PCA_Variance_Analysis.png")
plt.close()

# ============================================================================
# CREATE SUMMARY TABLE
# ============================================================================

summary_table = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(min(10, len(variance_explained)))],
    'Variance_%': variance_explained[:10],
    'Cumulative_%': cumulative_variance[:10],
    'Use_Case': [
        'Visualization (Primary)',
        'Visualization (Secondary)',
        'Better variance (90%)',
        'Better variance (90%)',
        'Better variance (90%)',
        'Marginal improvement',
        'Marginal improvement',
        'Minimal gain',
        'Minimal gain',
        'Minimal gain'
    ][:len(variance_explained[:10])]
})

summary_table.to_csv('PCA_Component_Summary.csv', index=False)
print("  ✓ Saved: PCA_Component_Summary.csv")
