import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folders automatically
os.makedirs("cleaned_data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/Messy_Employee_dataset.csv")

print("========== ORIGINAL DATA ==========")
print(data.head())

# Standardize column names
data.columns = data.columns.str.strip().str.replace(" ", "_")

# Missing values before cleaning
missing_values = data.isnull().sum()

print("\n========== MISSING VALUES ==========")
print(missing_values)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Fill missing numeric values
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_columns:
    data[col].fillna(data[col].mean(), inplace=True)

# Fill missing categorical values
categorical_columns = data.select_dtypes(include=['object']).columns

for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Clean phone numbers
if 'Phone' in data.columns:
    data['Phone'] = data['Phone'].astype(str)

# Save cleaned dataset
cleaned_file = "cleaned_data/cleaned_employee_dataset.csv"
data.to_csv(cleaned_file, index=False)

print("\nCleaned dataset saved successfully!")

# Statistical summary
summary = data.describe(include='all')

summary_file = "reports/summary_report.txt"

with open(summary_file, "w") as f:
    f.write(str(summary))

print("Summary report generated!")

# Missing values visualization
plt.figure(figsize=(10, 5))
missing_values.plot(kind='bar')

plt.title("Missing Values Before Cleaning")
plt.xlabel("Columns")
plt.ylabel("Count")

plt.tight_layout()

missing_chart = "reports/missing_values.png"
plt.savefig(missing_chart)

# Salary distribution
if 'Salary' in data.columns:
    plt.figure(figsize=(8, 5))

    sns.histplot(data['Salary'], kde=True)

    plt.title("Salary Distribution")

    salary_chart = "reports/salary_distribution.png"
    plt.savefig(salary_chart)

# Correlation heatmap
plt.figure(figsize=(8, 6))

numeric_data = data.select_dtypes(include=['float64', 'int64'])

sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

heatmap_chart = "reports/correlation_heatmap.png"
plt.savefig(heatmap_chart)

print("\nReports generated successfully!")

print("\n========== CLEANED DATA ==========")
print(data.head())
