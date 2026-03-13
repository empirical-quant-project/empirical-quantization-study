import pandas as pd
# Define the new file path
file_path = "/home/saima/projects/NonFunc-AWQ/scripts/statistical-tests/py-codellama13B-AWQ-all.csv"

# Load the CSV file
df = pd.read_csv(file_path)

# Convert all values to float where possible
for col in df.columns:
    try:
        df[col] = df[col].astype(float)
    except ValueError:
        print(f"Warning: Could not convert column '{col}' to float. It may contain non-numeric data.")

# Save the changes back to the same file
df.to_csv(file_path, index=False)

# Confirm completion
file_path
