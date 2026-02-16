"""
WORKSTREAM 1: COMPLETE ANALYSIS USING ALL SUBJECTS
Treats age as continuous variable (18-71 years)
Uses all 36 subjects instead of just young/old binary classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("WORKSTREAM 1: ANALYSIS WITH ALL SUBJECTS (CONTINUOUS AGE)")
print("="*80)

# Load data
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
    'Jaw (PD8, Step Loading)'
]

# Load all data
all_data = {}
for sheet in sheets:
    df = pd.read_excel(excel_file, sheet_name=sheet, engine='openpyxl')
    all_data[sheet] = df
    print(f"✓ Loaded {sheet}: {df.shape}")

print(f"\n✓ Successfully loaded {len(all_data)} experimental conditions")

# ============================================================================
# STEP 1: CREATE MASTER DATASET WITH ALL SUBJECTS
# ============================================================================
print("\n" + "="*80)
print("STEP 1: CREATE MASTER DATASET - ALL 36 SUBJECTS")
print("="*80)

# Use first sheet to get subject info
df_ref = all_data['Forehead (PD2, Step Loading)']

print(f"\nTotal subjects: {len(df_ref)}")
print(f"Age range: {df_ref['Age'].min()} - {df_ref['Age'].max()} years")
print(f"Mean age: {df_ref['Age'].mean():.1f} ± {df_ref['Age'].std():.1f} years")

# Age distribution
age_bins = [15, 25, 35, 45, 55, 65, 75]
age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75']
df_ref['AgeGroup'] = pd.cut(df_ref['Age'], bins=age_bins, labels=age_labels)

print("\nAge distribution:")
print(df_ref['AgeGroup'].value_counts().sort_index())

# Define Cutometer parameters
cutometer_params = [' R0', ' R1', ' R2', ' R3', ' R4', ' R5', ' R6', 
                    ' R7', ' R8', ' F0', ' F1', ' Q0', ' Q1', ' Q2', ' Q3']

# Create master dataset with ALL subjects
print("\nCombining all conditions...")
master_data = []

for sheet_name, df_sheet in all_data.items():
    # Parse condition from sheet name
    parts = sheet_name.replace('(', '').replace(')', '').split()
    location = parts[0]
    probe = parts[1].replace('PD', '')
    loading = parts[2]
    
    # Take ALL subjects (no filtering)
    df_all = df_sheet.copy()
    
    # Add condition metadata
    df_all['Location'] = location
    df_all['ProbeSize'] = probe
    df_all['LoadingMode'] = loading
    
    master_data.append(df_all)

# Combine all
master_df = pd.concat(master_data, ignore_index=True)

# Remove any rows with missing age
master_df = master_df[master_df['Age'].notna()]

print(f"\n✓ Master dataset created!")
print(f"  Total measurements: {len(master_df)}")
print(f"  Unique subjects: {master_df['Subjects'].nunique()}")
print(f"  Parameters per measurement: {len(cutometer_params)}")

# ============================================================================
# STEP 2: CORRELATION WITH AGE (CONTINUOUS)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CORRELATION ANALYSIS - PARAMETERS vs AGE")
print("="*80)

# Extract features and age
X = master_df[cutometer_params].values
age = master_df['Age'].values

# Handle missing values
X = np.nan_to_num(X, nan=np.nanmedian(X))

# Parameter names
param_names = [p.strip() for p in cutometer_params]

# Calculate Pearson correlation with age for each parameter
print("\nCalculating correlations between each parameter and age...")
age_correlations = []

for i, param in enumerate(param_names):
    # Get values for this parameter
    param_values = X[:, i]
    
    # Calculate Pearson correlation with age
    r, p_value = stats.pearsonr(param_values, age)
    
    # Calculate Spearman correlation (non-parametric)
    r_spearman, p_spearman = stats.spearmanr(param_values, age)
    
    age_correlations.append({
        'Parameter': param,
        'Pearson_r': r,
        'Pearson_p': p_value,
        'Spearman_r': r_spearman,
        'Spearman_p': p_spearman,
        'Significant_Pearson': p_value < 0.05,
        'Significant_Spearman': p_spearman < 0.05,
        'Direction': 'Increases' if r > 0 else 'Decreases'
    })

age_corr_df = pd.DataFrame(age_correlations).sort_values('Pearson_r', key=abs, ascending=False)

print("\nCorrelation with Age (sorted by strength):")
print("="*100)
print(f"{'Parameter':<10} {'Pearson r':<12} {'P-value':<12} {'Spearman ρ':<12} {'P-value':<12} {'Direction':<12}")
print("="*100)

for _, row in age_corr_df.iterrows():
    sig_p = "***" if row['Pearson_p'] < 0.001 else ("**" if row['Pearson_p'] < 0.01 else ("*" if row['Pearson_p'] < 0.05 else "ns"))
    sig_s = "***" if row['Spearman_p'] < 0.001 else ("**" if row['Spearman_p'] < 0.01 else ("*" if row['Spearman_p'] < 0.05 else "ns"))
    print(f"{row['Parameter']:<10} {row['Pearson_r']:>8.3f} {sig_p:<3} {row['Pearson_p']:>8.4f}  {row['Spearman_r']:>8.3f} {sig_s:<3} {row['Spearman_p']:>8.4f}  {row['Direction']:<12}")

n_sig_pearson = len(age_corr_df[age_corr_df['Significant_Pearson']==True])
n_sig_spearman = len(age_corr_df[age_corr_df['Significant_Spearman']==True])

print(f"\n✓ {n_sig_pearson} parameters significantly correlated with age (Pearson, p<0.05)")
print(f"✓ {n_sig_spearman} parameters significantly correlated with age (Spearman, p<0.05)")

# ============================================================================
# STEP 3: LINEAR REGRESSION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: LINEAR REGRESSION - PREDICTING AGE FROM PARAMETERS")
print("="*80)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit linear regression for each parameter individually
print("\nTesting each parameter as age predictor...")
regression_results = []

for i, param in enumerate(param_names):
    # Single parameter regression
    X_single = X_scaled[:, i].reshape(-1, 1)
    
    lr = LinearRegression()
    lr.fit(X_single, age)
    
    # R-squared (coefficient of determination)
    r2 = lr.score(X_single, age)
    
    # Slope (coefficient)
    slope = lr.coef_[0]
    
    regression_results.append({
        'Parameter': param,
        'R_squared': r2,
        'Slope': slope,
        'Variance_Explained_%': r2 * 100
    })

regression_df = pd.DataFrame(regression_results).sort_values('R_squared', ascending=False)

print("\nLinear Regression Results (R²):")
print("="*70)
print(f"{'Parameter':<10} {'R²':<10} {'Variance Explained':<20} {'Slope':<10}")
print("="*70)

for _, row in regression_df.iterrows():
    print(f"{row['Parameter']:<10} {row['R_squared']:<10.4f} {row['Variance_Explained_%']:<18.2f}% {row['Slope']:<10.3f}")

# Multi-parameter regression (using all parameters)
print("\nMulti-parameter model (all 15 parameters):")
lr_multi = LinearRegression()
lr_multi.fit(X_scaled, age)
r2_multi = lr_multi.score(X_scaled, age)
print(f"  R² = {r2_multi:.4f} ({r2_multi*100:.2f}% variance explained)")

# Cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(lr_multi, X_scaled, age, cv=5, scoring='r2')
print(f"  Cross-validated R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================================
# STEP 4: PARAMETER CORRELATION MATRIX (SAME AS BEFORE)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: PARAMETER CORRELATION MATRIX")
print("="*80)

corr_matrix = np.corrcoef(X.T)

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(param_names)):
    for j in range(i+1, len(param_names)):
        if abs(corr_matrix[i,j]) > 0.8:
            high_corr_pairs.append((param_names[i], param_names[j], corr_matrix[i,j]))

high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"\nHighly correlated parameter pairs (|r| > 0.8): {len(high_corr_pairs)}")
for p1, p2, corr in high_corr_pairs:
    print(f"  {p1} <-> {p2}: r = {corr:.3f}")

# ============================================================================
# STEP 5: PCA WITH ALL SUBJECTS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: PRINCIPAL COMPONENT ANALYSIS")
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

# ============================================================================
# STEP 6: MUTUAL INFORMATION (FOR CONTINUOUS AGE)
# ============================================================================
print("\n" + "="*80)
print("STEP 6: MUTUAL INFORMATION WITH AGE")
print("="*80)

mi_scores = mutual_info_regression(X_scaled, age, random_state=42)
mi_df = pd.DataFrame({
    'Parameter': param_names,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("\nMutual Information Scores:")
print(mi_df.to_string(index=False))

# ============================================================================
# STEP 7: RANDOM FOREST REGRESSION
# ============================================================================
print("\n" + "="*80)
print("STEP 7: RANDOM FOREST FEATURE IMPORTANCE (REGRESSION)")
print("="*80)

rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_scaled, age)

# R² score
r2_rf = rf.score(X_scaled, age)
print(f"\nRandom Forest R²: {r2_rf:.4f}")

# Cross-validation
cv_scores_rf = cross_val_score(rf, X_scaled, age, cv=5, scoring='r2')
print(f"Cross-validated R²: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

# Feature importance
rf_importance = rf.feature_importances_
rf_df = pd.DataFrame({
    'Parameter': param_names,
    'Importance': rf_importance
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_df.to_string(index=False))

# ============================================================================
# STEP 8: COMBINED RANKING (CONTINUOUS AGE VERSION)
# ============================================================================
print("\n" + "="*80)
print("STEP 8: COMBINED RANKING FOR CONTINUOUS AGE")
print("="*80)

# Normalize scores
def normalize_rank(values):
    values = np.array(values)
    return (values - values.min()) / (values.max() - values.min())

# Create combined DataFrame
combined_df = pd.DataFrame({'Parameter': param_names})

# Add correlation scores
combined_df = combined_df.merge(age_corr_df[['Parameter', 'Pearson_r', 'Pearson_p']], on='Parameter')
combined_df['Correlation_Score'] = normalize_rank(np.abs(combined_df['Pearson_r']))
combined_df['Significant'] = combined_df['Pearson_p'] < 0.05

# Add MI scores
combined_df = combined_df.merge(mi_df, on='Parameter')
combined_df['MI_Normalized'] = normalize_rank(combined_df['MI_Score'])

# Add RF scores
combined_df = combined_df.merge(rf_df, on='Parameter')
combined_df['RF_Normalized'] = normalize_rank(combined_df['Importance'])

# Redundancy (from correlation matrix)
redundancy_counts = []
for i, param in enumerate(param_names):
    count = sum(1 for j in range(len(param_names)) if i != j and abs(corr_matrix[i,j]) > 0.8)
    redundancy_counts.append(count)

combined_df['Redundancy_Count'] = redundancy_counts
combined_df['Uniqueness_Score'] = 1 - (combined_df['Redundancy_Count'] / max(redundancy_counts)) if max(redundancy_counts) > 0 else 1

# Calculate combined score
combined_df['Combined_Score'] = (
    combined_df['Correlation_Score'] * 0.35 +
    combined_df['MI_Normalized'] * 0.35 +
    combined_df['RF_Normalized'] * 0.20 +
    combined_df['Uniqueness_Score'] * 0.10
)

combined_df = combined_df.sort_values('Combined_Score', ascending=False)

print("\nCombined Parameter Ranking:")
print("="*120)
print(f"{'Rank':<6} {'Parameter':<10} {'Combined':<10} {'Corr|r|':<10} {'MI':<10} {'RF':<10} {'Unique':<8} {'Redundancy':<12}")
print("="*120)

for rank, (_, row) in enumerate(combined_df.iterrows(), 1):
    sig_marker = "***" if row['Pearson_p'] < 0.001 else ("**" if row['Pearson_p'] < 0.01 else ("*" if row['Pearson_p'] < 0.05 else ""))
    print(f"{rank:<6} {row['Parameter']:<10} {row['Combined_Score']:<10.3f} "
          f"{abs(row['Pearson_r']):<7.3f}{sig_marker:<3} {row['MI_Score']:<10.3f} "
          f"{row['Importance']:<10.3f} {row['Uniqueness_Score']:<8.3f} {row['Redundancy_Count']:<12}")

print("\n" + "="*80)
print("DETERMINING MINIMAL SET")
print("="*80)

# Select top parameters (threshold: combined score > 0.6 or top 7)
top_threshold = 0.60
top_params = combined_df[combined_df['Combined_Score'] >= top_threshold]

if len(top_params) < 5:
    # If too few, take top 7
    top_params = combined_df.head(7)
    
optimal_k = len(top_params)

print(f"\nRecommended minimal set: {optimal_k} parameters")
print(f"Selection criteria: Combined score ≥ {top_threshold}")

print(f"\nThe {optimal_k} most important parameters:")
for i, (_, row) in enumerate(top_params.iterrows(), 1):
    print(f"  {i}. {row['Parameter']:<6} (score: {row['Combined_Score']:.3f}, "
          f"r = {row['Pearson_r']:>6.3f}, p = {row['Pearson_p']:.4f})")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 9: SAVING RESULTS")
print("="*80)

import os
if not os.path.exists('results'):
    os.makedirs('results')

with pd.ExcelWriter('results/workstream1_ALL_SUBJECTS.xlsx', engine='openpyxl') as writer:
    combined_df.to_excel(writer, sheet_name='Parameter_Rankings', index=False)
    age_corr_df.to_excel(writer, sheet_name='Age_Correlations', index=False)
    regression_df.to_excel(writer, sheet_name='Linear_Regression', index=False)
    mi_df.to_excel(writer, sheet_name='Mutual_Information', index=False)
    rf_df.to_excel(writer, sheet_name='Random_Forest', index=False)
    
    # Summary
    summary_data = {
        'Metric': [
            'Total Subjects',
            'Age Range',
            'Mean Age (years)',
            'Total Measurements',
            'Parameters with Significant Age Correlation (p<0.05)',
            'PCs for 90% Variance',
            'Recommended Minimal Set',
            'Multi-parameter R²',
            'Random Forest R²'
        ],
        'Value': [
            master_df['Subjects'].nunique(),
            f"{master_df['Age'].min():.0f} - {master_df['Age'].max():.0f}",
            f"{master_df['Age'].mean():.1f} ± {master_df['Age'].std():.1f}",
            len(master_df),
            n_sig_pearson,
            n_components_90,
            optimal_k,
            f"{r2_multi:.4f}",
            f"{r2_rf:.4f}"
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

print("✓ Saved: workstream1_ALL_SUBJECTS.xlsx")

# ============================================================================
# STEP 10: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 10: CREATING VISUALIZATIONS")
print("="*80)

# 1. Age distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(master_df['Age'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(master_df['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {master_df["Age"].mean():.1f}')
ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Age Distribution - All 36 Subjects', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/age_distribution_all_subjects.png', dpi=300)
print("✓ Saved: age_distribution_all_subjects.png")
plt.close()

# 2. Top parameters correlation with age
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

top_8_params = combined_df.head(8)

for idx, (_, row) in enumerate(top_8_params.iterrows()):
    ax = axes[idx]
    param = row['Parameter']
    
    # Get data
    param_col = f' {param}'
    param_idx = param_names.index(param)
    param_values = X[:, param_idx]
    
    # Scatter plot
    ax.scatter(age, param_values, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # Regression line
    z = np.polyfit(age, param_values, 1)
    p = np.poly1d(z)
    age_line = np.linspace(age.min(), age.max(), 100)
    ax.plot(age_line, p(age_line), 'r-', linewidth=2, label='Linear fit')
    
    # Labels
    ax.set_xlabel('Age (years)', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'{param} Value', fontsize=10, fontweight='bold')
    ax.set_title(f'{param}\nr = {row["Pearson_r"]:.3f}, p = {row["Pearson_p"]:.4f}', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('Top 8 Parameters: Correlation with Age (All Subjects)', 
            fontsize=16, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig('results/top_parameters_vs_age_all_subjects.png', dpi=300)
print("✓ Saved: top_parameters_vs_age_all_subjects.png")
plt.close()

# 3. Combined ranking visualization
fig, ax = plt.subplots(figsize=(12, 8))

top_10 = combined_df.head(10)
colors = ['gold' if row['Significant'] and row['Redundancy_Count'] <= 1 else 'lightgreen' if row['Significant'] else 'lightcoral' 
          for _, row in top_10.iterrows()]

y_pos = range(len(top_10))
bars = ax.barh(y_pos, top_10['Combined_Score'], color=colors, alpha=0.8, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_10['Parameter'])
ax.set_xlabel('Combined Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Parameters - Continuous Age Analysis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (_, row) in enumerate(top_10.iterrows()):
    ax.text(row['Combined_Score'] + 0.01, i, f"{row['Combined_Score']:.3f}", 
            va='center', fontsize=10, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gold', alpha=0.8, label='⭐ Top: Significant & Unique'),
    Patch(facecolor='lightgreen', alpha=0.8, label='✓ Significant'),
    Patch(facecolor='lightcoral', alpha=0.8, label='Not significant')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('results/parameter_ranking_all_subjects.png', dpi=300)
print("✓ Saved: parameter_ranking_all_subjects.png")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE - ALL SUBJECTS")
print("="*80)

print(f"\nKey Findings:")
print(f"  • Used all {master_df['Subjects'].nunique()} subjects (age range: {master_df['Age'].min():.0f}-{master_df['Age'].max():.0f} years)")
print(f"  • {n_sig_pearson} parameters significantly correlated with age")
print(f"  • Multi-parameter model R² = {r2_multi:.4f} ({r2_multi*100:.1f}% variance explained)")
print(f"  • Recommended minimal set: {optimal_k} parameters")

print(f"\nTop {optimal_k} Parameters:")
for i, (_, row) in enumerate(top_params.iterrows(), 1):
    print(f"  {i}. {row['Parameter']} (r = {row['Pearson_r']:.3f}, p = {row['Pearson_p']:.4f})")

print("\n" + "="*80)