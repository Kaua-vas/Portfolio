import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Paths to the test file and store information (relative paths)
test_file_path = './test.csv'
store_file_path = './store.csv'

# Load the test dataset
test_df = pd.read_csv(test_file_path)

# Fill missing values in the 'Open' column with 1 (assuming the store was open)
test_df['Open'].fillna(1, inplace=True)

# Save the cleaned test file to ensure consistency
test_cleaned_file_path = './test_cleaned.csv'
test_df.to_csv(test_cleaned_file_path, index=False)

print("Test file cleaned and saved successfully!")

# Load the store information to merge with the test data
store_df = pd.read_csv(store_file_path)

# Merge the cleaned test data with store information
test_df_cleaned = pd.read_csv(test_cleaned_file_path)
test_df_merged = pd.merge(test_df_cleaned, store_df, on='Store', how='left')

# Generate dummies for categorical variables in the test set
categorical_columns = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
test_df_merged = pd.get_dummies(test_df_merged, columns=categorical_columns, drop_first=True)

# Load the previously saved scaler to apply normalization (relative path)
scaler_path = './scaler_previsao_vendas.save'
scaler = joblib.load(scaler_path)

# Ensure that the test set has the same columns as the training set (based on the scaler)
scaler_columns = scaler.feature_names_in_

# Add missing columns in the test set (if any) and fill with 0
for col in scaler_columns:
    if col not in test_df_merged.columns:
        test_df_merged[col] = 0

# Remove extra columns not seen by the scaler
test_df_cleaned_final = test_df_merged[scaler_columns]

# Apply normalization to the test set
X_test_scaled = scaler.transform(test_df_cleaned_final)

# Load the trained model for prediction (relative path)
model_path = './modelo_previsao_vendas_final.keras'
modelo_salvo = load_model(model_path)

# Make predictions on the scaled test set
previsoes = modelo_salvo.predict(X_test_scaled)

# Save predictions in a new CSV file for submission
output_file_path = './submission_updated.csv'
output_df = pd.DataFrame({'Id': test_df_cleaned['Id'], 'Sales': previsoes.flatten()})

# Replace NaN values with 0 in the predictions (if any)
output_df['Sales'].fillna(0, inplace=True)

# Save the final corrected submission file (relative path)
output_file_path_corrected = './submission_corrected.csv'
output_df.to_csv(output_file_path_corrected, index=False)

print(f"Predictions saved with corrections in {output_file_path_corrected}")
