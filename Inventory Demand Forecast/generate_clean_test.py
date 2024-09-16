import pandas as pd

# Path to the original test file (relative path)
test_file_path = './test.csv'

# Load the test dataset
test_df = pd.read_csv(test_file_path)

# Fill missing values in the 'Open' column with 1 (assuming the store was open)
test_df['Open'].fillna(1, inplace=True)

# Save the cleaned test file (relative path)
test_cleaned_file_path = './test_cleaned.csv'
test_df.to_csv(test_cleaned_file_path, index=False)

print("Test file cleaned and saved successfully!")
