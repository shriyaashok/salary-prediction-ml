import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 10,000 data points
n_samples = 10000

# Generate experience (0-20 years, weighted towards lower values)
# Probabilities for 0-20 years (21 values)
exp_probs = [0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.05, 
             0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]
# Normalize to sum to 1
exp_probs = np.array(exp_probs) / np.sum(exp_probs)
experience = np.random.choice(range(0, 21), size=n_samples, p=exp_probs)

# Generate test scores (1-10, weighted towards middle-high values)
test_probs = [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.15, 0.12]
test_probs = np.array(test_probs) / np.sum(test_probs)
test_score = np.random.choice(range(1, 11), size=n_samples, p=test_probs)

# Generate interview scores (1-10, weighted towards middle-high values)
interview_probs = [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.15, 0.12]
interview_probs = np.array(interview_probs) / np.sum(interview_probs)
interview_score = np.random.choice(range(1, 11), size=n_samples, p=interview_probs)

# Calculate base salary in INR (50k to 3 lakh range)
# Base salary: 50k (50,000) + experience bonus
base_salary_inr = 50000 + (experience * 10000)  # Each year adds 10k

# Add bonuses based on test score (higher test score = higher bonus)
test_bonus = (test_score - 5) * 5000  # Range: -20k to +25k

# Add bonuses based on interview score (higher interview score = higher bonus)
interview_bonus = (interview_score - 5) * 6000  # Range: -24k to +30k

# Calculate final salary with some randomness
salary = base_salary_inr + test_bonus + interview_bonus + np.random.normal(0, 15000, n_samples)

# Ensure salary is within realistic INR bounds (50k - 3 lakh)
salary = np.clip(salary, 50000, 300000)

# Round to nearest 1000 for cleaner numbers
salary = np.round(salary / 1000) * 1000

# Create DataFrame
data = pd.DataFrame({
    'experience': experience,
    'test_score': test_score,
    'interview_score': interview_score,
    'salary': salary.astype(int)
})

# Sort by experience, then by salary for better organization
data = data.sort_values(['experience', 'salary']).reset_index(drop=True)

# Save to CSV
data.to_csv('hiring.csv', index=False)

print(f"Generated {len(data)} samples")
print(f"\nDataset Statistics:")
print(data.describe())
print(f"\nSalary range: ${data['salary'].min():,} - ${data['salary'].max():,}")
print(f"\nSample data:")
print(data.head(10))
print(f"\nSaved to hiring.csv")

