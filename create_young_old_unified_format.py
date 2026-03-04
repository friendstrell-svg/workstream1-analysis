import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("="*80)
print("CREATING YOUNG VS OLD PCA DATA IN UNIFIED FORMAT")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

input_file = 'data.xlsx'
output_file = 'PCA_Young_vs_Old_UNIFIED_FORMAT.xlsx'

YOUNG_THRESHOLD = 35
OLD_THRESHOLD = 50

param_cols = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 
              'F0', 'F1', 'Q0', 'Q1', 'Q2', 'Q3']

# Sheet configurations
sheet_configs = [
    ('Forehead (PD2, Step Loading)', 'Forehead', 2, 'Step'),
    ('Forehead (PD2, Ramp Loading)', 'Forehead', 2, 'Ramp'),
    ('Forehead (PD8, Step Loading)', 'Forehead', 8, 'Step'),
    ('Forehead (PD8, Ramp Loading)', 'Forehead', 8, 'Ramp'),
    ('Parotid (PD2, Step Loading)', 'Parotid', 2, 'Step'),
    ('Parotid (PD2, Ramp Loading)', 'Parotid', 2, 'Ramp'),
    ('Parotid (PD8, Step Loading)', 'Parotid', 8, 'Step'),
    ('Parotid (PD8, Ramp Loading)', 'Parotid', 8, 'Ramp'),
    ('Jaw (PD2, Step Loading)', 'Jaw', 2, 'Step'),
    ('Jaw (PD2, Ramp Loading)', 'Jaw', 2, 'Ramp'),
    ('Jaw (PD8, Step Loading)', 'Jaw', 8, 'Step'),
    ('Jaw (PD8, Ramp Loading)', 'Jaw', 8, 'Ramp')
]

# ============================================================================
# LOAD AND PROCESS ALL DATA
# ============================================================================

print("\nStep 1: Loading and processing all conditions...")

all_data = []

for sheet_name, location, probe_dia, loading in sheet_configs:
    print(f"\n  Processing: {sheet_name}")
    
    try:
        # Load sheet
        df = pd.read_excel(input_file, sheet_name=sheet_name, engine='openpyxl')
        df.columns = df.columns.str.strip()
        
        # Find age column
        age_col = 'Age' if 'Age' in df.columns else None
        if not age_col:
            print(f"    ⚠ No age column - skipping")
            continue
        
        # Find available parameters
        available_params = [col for col in param_cols if col in df.columns]
        if len(available_params) < len(param_cols):
            available_params = [col if col in df.columns else f' {col}' 
                              for col in param_cols 
                              if col in df.columns or f' {col}' in df.columns]
        
        # Find subject column
        subject_col = None
        for col in df.columns:
            if 'subject' in col.lower():
                subject_col = col
                break
        
        if not subject_col:
            df['Subject Number'] = range(1, len(df) + 1)
            subject_col = 'Subject Number'
        
        # Split into Young and Old
        df_young = df[df[age_col] <= YOUNG_THRESHOLD].copy()
        df_old = df[df[age_col] >= OLD_THRESHOLD].copy()
        
        # Process Young group
        if len(df_young) >= 3:
            X_young = df_young[available_params].values
            X_young_clean = np.nan_to_num(X_young, nan=np.nanmedian(X_young))
            scaler_young = StandardScaler()
            X_young_scaled = scaler_young.fit_transform(X_young_clean)
            pca_young = PCA(n_components=5)
            X_young_pca = pca_young.fit_transform(X_young_scaled)
            
            # Add to dataframe
            df_young['Age Group'] = 'Young'
            for i in range(5):
                df_young[f'PC{i+1}'] = X_young_pca[:, i]
        
        # Process Old group
        if len(df_old) >= 3:
            X_old = df_old[available_params].values
            X_old_clean = np.nan_to_num(X_old, nan=np.nanmedian(X_old))
            scaler_old = StandardScaler()
            X_old_scaled = scaler_old.fit_transform(X_old_clean)
            pca_old = PCA(n_components=5)
            X_old_pca = pca_old.fit_transform(X_old_scaled)
            
            # Add to dataframe
            df_old['Age Group'] = 'Old'
            for i in range(5):
                df_old[f'PC{i+1}'] = X_old_pca[:, i]
        
        # Combine young and old for this condition
        df_combined = pd.concat([df_young, df_old], ignore_index=True)
        
        # Add metadata columns
        df_combined['Facial Location'] = location
        df_combined['Probe Diameter (mm)'] = probe_dia
        df_combined['Loading Mode'] = loading
        
        # Rename subject column if needed
        if subject_col != 'Subject Number':
            df_combined['Subject Number'] = df_combined[subject_col]
        
        all_data.append(df_combined)
        
        print(f"    ✓ Young: {len(df_young)} subjects")
        print(f"    ✓ Old: {len(df_old)} subjects")
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)[:80]}")
        continue

# ============================================================================
# COMBINE ALL DATA
# ============================================================================

print("\n" + "="*80)
print("Step 2: Creating unified dataset...")
print("="*80)

if all_data:
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns to match data_with_ALL_PCs.xlsx format
    column_order = ['Subject Number', 'Age Group', 'Age', 'Facial Location', 
                   'Probe Diameter (mm)', 'Loading Mode']
    
    # Add parameters (strip spaces from column names)
    for param in param_cols:
        if param in df_all.columns:
            column_order.append(param)
        elif f' {param}' in df_all.columns:
            df_all[param] = df_all[f' {param}']
            column_order.append(param)
    
    # Add PC columns
    for i in range(1, 6):
        if f'PC{i}' in df_all.columns:
            column_order.append(f'PC{i}')
    
    # Select and reorder columns
    available_cols = [col for col in column_order if col in df_all.columns]
    df_final = df_all[available_cols].copy()
    
    # Sort by Subject Number, then Condition
    df_final = df_final.sort_values(['Subject Number', 'Facial Location', 
                                     'Probe Diameter (mm)', 'Loading Mode'])
    
    print(f"\n  Total rows: {len(df_final)}")
    print(f"  Young subjects: {len(df_final[df_final['Age Group'] == 'Young']['Subject Number'].unique())}")
    print(f"  Old subjects: {len(df_final[df_final['Age Group'] == 'Old']['Subject Number'].unique())}")
    print(f"  Conditions: {len(df_final.groupby(['Facial Location', 'Probe Diameter (mm)', 'Loading Mode']))}")

# ============================================================================
# CREATE SEPARATE VIEWS
# ============================================================================

print("\n" + "="*80)
print("Step 3: Creating separate views...")
print("="*80)

# Create Young-only view
df_young_only = df_final[df_final['Age Group'] == 'Young'].copy()
print(f"\n  Young-only view: {len(df_young_only)} rows")

# Create Old-only view
df_old_only = df_final[df_final['Age Group'] == 'Old'].copy()
print(f"  Old-only view: {len(df_old_only)} rows")

# Create pivot table by condition
pivot_data = []
for (location, probe, loading), group in df_final.groupby(['Facial Location', 'Probe Diameter (mm)', 'Loading Mode']):
    young_group = group[group['Age Group'] == 'Young']
    old_group = group[group['Age Group'] == 'Old']
    
    pivot_data.append({
        'Facial Location': location,
        'Probe Diameter (mm)': probe,
        'Loading Mode': loading,
        'Young_Count': len(young_group),
        'Old_Count': len(old_group),
        'Young_Age_Mean': young_group['Age'].mean() if len(young_group) > 0 else None,
        'Old_Age_Mean': old_group['Age'].mean() if len(old_group) > 0 else None,
        'Young_Age_Range': f"{young_group['Age'].min():.0f}-{young_group['Age'].max():.0f}" if len(young_group) > 0 else None,
        'Old_Age_Range': f"{old_group['Age'].min():.0f}-{old_group['Age'].max():.0f}" if len(old_group) > 0 else None
    })

df_pivot = pd.DataFrame(pivot_data)
print(f"  Pivot summary: {len(df_pivot)} conditions")

# ============================================================================
# SAVE TO EXCEL
# ============================================================================

print("\n" + "="*80)
print("Step 4: Saving to Excel...")
print("="*80)

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    # Sheet 1: All data combined
    df_final.to_excel(writer, sheet_name='All Data (Young & Old)', index=False)
    print("  ✓ Saved: All Data (Young & Old)")
    
    # Sheet 2: Young only
    df_young_only.to_excel(writer, sheet_name='Young Only', index=False)
    print("  ✓ Saved: Young Only")
    
    # Sheet 3: Old only
    df_old_only.to_excel(writer, sheet_name='Old Only', index=False)
    print("  ✓ Saved: Old Only")
    
    # Sheet 4: Pivot summary
    df_pivot.to_excel(writer, sheet_name='Summary by Condition', index=False)
    print("  ✓ Saved: Summary by Condition")
    
    # Sheet 5-16: Individual conditions with Young/Old combined
    for (location, probe, loading), group in df_final.groupby(['Facial Location', 'Probe Diameter (mm)', 'Loading Mode']):
        sheet_name = f"{location[:3]}_{int(probe)}mm_{loading[:4]}"
        group.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  ✓ Saved: {sheet_name}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)

print(f"""
OUTPUT FILE: {output_file}

SHEETS CREATED:

1. All Data (Young & Old)
   • All subjects from all conditions
   • Columns: Subject Number, Age Group, Age, Location, Probe, Loading, 
             R0-R8, F0-F1, Q0-Q3, PC1-PC5
   • {len(df_final)} total rows
   • Easy to filter by Age Group!

2. Young Only
   • Only young subjects (Age ≤ {YOUNG_THRESHOLD})
   • Same column structure
   • {len(df_young_only)} rows

3. Old Only
   • Only old subjects (Age ≥ {OLD_THRESHOLD})
   • Same column structure
   • {len(df_old_only)} rows

4. Summary by Condition
   • Counts and age ranges for each condition
   • Quick overview

5-16. Individual condition sheets
   • One sheet per condition
   • Young and Old combined
   • Easy to see both groups side-by-side

COLUMN STRUCTURE (matches data_with_ALL_PCs.xlsx):
  Subject Number | Age Group | Age | Facial Location | Probe Diameter (mm) | 
  Loading Mode | R0 | R1 | ... | Q3 | PC1 | PC2 | PC3 | PC4 | PC5

KEY FEATURES:
  ✓ Subject numbers visible
  ✓ Age Group column (Young/Old)
  ✓ Age column (actual age)
  ✓ All parameters included
  ✓ PC1-PC5 scores included
  ✓ Easy to filter and sort
  ✓ Same format as data_with_ALL_PCs.xlsx
""")

print("="*80)
print("File ready for easy viewing and sharing!")
print("="*80)
