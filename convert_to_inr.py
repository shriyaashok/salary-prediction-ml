import pandas as pd

# Read the CSV file
df = pd.read_csv('hiring.csv')

# Convert USD to INR (1 USD = 83 INR)
USD_TO_INR = 83.0
df['salary'] = (df['salary'] * USD_TO_INR).astype(int)

# Save back to CSV
df.to_csv('hiring.csv', index=False)

print(f'Converted {len(df)} salaries from USD to INR')
print(f'Salary range: INR {df["salary"].min():,} - INR {df["salary"].max():,}')
print(f'\nSample data:')
print(df.head(10))

