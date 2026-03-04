import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

print("="*80)
print("CREATING YOUR PCA FILES WITH PC1/PC2 SCORES")
print("="*80)

# ============================================================================
# STEP 1: LOAD YOUR DATA
# ============================================================================
print("\nStep 1: Loading your data...")

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

# Cutometer parameters
cutometer_params = [' R0', ' R1', ' R2', ' R3', ' R4', ' R5', ' R6', 
                    ' R7', ' R8', ' F0', ' F1', ' Q0', ' Q1', ' Q2', ' Q3']

# Load all sheets
all_data = {}
for sheet in sheets:
    df = pd.read_excel(excel_file, sheet_name=sheet, engine='openpyxl')
    all_data[sheet] = df
    print(f"  ✓ {sheet}: {df.shape}")

print("\n" + "="*80)
print("Step 2: Performing PCA for each experimental condition")
print("="*80)

# Create output directory
if not os.path.exists('results/PCA_Files'):
    os.makedirs('results/PCA_Files')

for sheet_name, df_sheet in all_data.items():
    print(f"\n--- Processing: {sheet_name} ---")
    
    # Extract features (15 parameters)
    X = df_sheet[cutometer_params].values
    
    # Handle missing values
    X_clean = np.nan_to_num(X, nan=np.nanmedian(X))
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Perform PCA
    pca = PCA(n_components=2)  # Only need PC1 and PC2
    X_pca = pca.fit_transform(X_scaled)
    
    # Get PC1 and PC2 scores
    PC1_scores = X_pca[:, 0]
    PC2_scores = X_pca[:, 1]
    
    print(f"  PCA variance explained:")
    print(f"    PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"    PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"    Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")

    # Create output dataframe
    output_df = pd.DataFrame()
    
    # Add subject numbers
    output_df['Subjects'] = df_sheet['Subjects']
    
    # Add all 15 parameters
    for param in cutometer_params:
        output_df[param.strip()] = df_sheet[param]
    
    # Add age
    output_df['Age'] = df_sheet['Age']
    
    # Add PC scores
    output_df['PC1'] = PC1_scores
    output_df['PC2'] = PC2_scores

    
    # Create filename matching uploaded file format
    # "Forehead (PD2, Step Loading)" → "Forehead__PD2__Step_Loading__YOUR.csv"
    
    parts = sheet_name.replace('(', '').replace(')', '').replace(',', '').split()
    location = parts[0]  # Forehead, Parotid, or Jaw
    probe = parts[1]     # PD2 or PD8
    loading = ' '.join(parts[2:])  # Step Loading or Ramp Loading
    
    filename = f"{location}__{probe}__{loading.replace(' ', '_')}__YOUR.csv"
    filepath = f"results/PCA_Files/{filename}"
    
    # Save to CSV
    output_df.to_csv(filepath, index=False)
    print(f"  ✓ Saved: {filename}")


print("\n" + "="*80)
print("Step 5: Creating comparison summary")
print("="*80)

# Create a summary comparing variance explained across conditions
summary_data = []

for sheet_name, df_sheet in all_data.items():
    # Extract and process data
    X = df_sheet[cutometer_params].values
    X_clean = np.nan_to_num(X, nan=np.nanmedian(X))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # PCA
    pca = PCA(n_components=5)
    pca.fit(X_scaled)
    
    # Store results
    parts = sheet_name.replace('(', '').replace(')', '').replace(',', '').split()
    location = parts[0]
    probe = parts[1]
    loading = ' '.join(parts[2:])
    
    summary_data.append({
        'Location': location,
        'Probe': probe,
        'Loading': loading,
        'PC1_Variance_%': pca.explained_variance_ratio_[0] * 100,
        'PC2_Variance_%': pca.explained_variance_ratio_[1] * 100,
        'PC1+PC2_Total_%': sum(pca.explained_variance_ratio_[:2]) * 100,
        'PC1-5_Total_%': sum(pca.explained_variance_ratio_) * 100
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results/PCA_Files/PCA_Variance_Summary.csv', index=False)
print("✓ Saved: PCA_Variance_Summary.csv")


print(f"""
CREATED 12 CSV FILES:

Location: results/PCA_Files/

Format (matching uploaded files):
  Columns: Subjects, R0-R8, F0-F1, Q0-Q3, Age, PC1, PC2

Files created:
  1. Forehead__PD2__Step_Loading__YOUR.csv
  2. Forehead__PD2__Ramp_Loading__YOUR.csv
  3. Forehead__PD8__Step_Loading__YOUR.csv
  4. Forehead__PD8__Ramp_Loading__YOUR.csv
  5. Parotid__PD2__Step_Loading__YOUR.csv
  6. Parotid__PD2__Ramp_Loading__YOUR.csv
  7. Parotid__PD8__Step_Loading__YOUR.csv
  8. Parotid__PD8__Ramp_Loading__YOUR.csv
  9. Jaw__PD2__Step_Loading__YOUR.csv
  10. Jaw__PD2__Ramp_Loading__YOUR.csv
  11. Jaw__PD8__Step_Loading__YOUR.csv
  12. Jaw__PD8__Ramp_Loading__YOUR.csv

PLUS:
  PCA_Variance_Summary.csv (comparison across all conditions)

""")
