import pandas as pd

excel_file = 'data.xlsx'
df = pd.read_excel(excel_file, sheet_name='Forehead (PD2, Step Loading)', engine='openpyxl')

print("="*60)
print("SUBJECT COUNT VERIFICATION")
print("="*60)

print("\nAge distribution:")
print(f"Young (≤35): {len(df[df['Age'] <= 35])} subjects")
print(f"Middle (36-49): {len(df[(df['Age'] > 35) & (df['Age'] < 50)])} subjects")
print(f"Old (≥50): {len(df[df['Age'] >= 50])} subjects")
print(f"Total: {len(df)} subjects")

print("\n" + "="*60)
print("OLD SUBJECTS (≥50):")
print("="*60)
old_df = df[df['Age'] >= 50][['Subjects', 'Age']].sort_values('Age')
for _, row in old_df.iterrows():
    print(f"Subject {int(row['Subjects'])}: Age {int(row['Age'])}")

print("\n" + "="*60)
print("YOUNG SUBJECTS (≤35):")
print("="*60)
young_df = df[df['Age'] <= 35][['Subjects', 'Age']].sort_values('Age')
for _, row in young_df.iterrows():
    print(f"Subject {int(row['Subjects'])}: Age {int(row['Age'])}")

print("\n" + "="*60)
print("MIDDLE SUBJECTS (36-49):")
print("="*60)
middle_df = df[(df['Age'] > 35) & (df['Age'] < 50)][['Subjects', 'Age']].sort_values('Age')
for _, row in middle_df.iterrows():
    print(f"Subject {int(row['Subjects'])}: Age {int(row['Age'])}")