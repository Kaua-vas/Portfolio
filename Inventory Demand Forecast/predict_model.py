import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load historical data to compare with predictions
train_file_path = './train.csv'
train_df = pd.read_csv(train_file_path)

# Merge the historical sales with the store info to get comparable data
historical_sales = pd.merge(train_df, store_df, on='Store', how='left')

# Convert 'Date' to datetime in both datasets
historical_sales['Date'] = pd.to_datetime(historical_sales['Date'])
test_df_cleaned['Date'] = pd.to_datetime(test_df_cleaned['Date'])

# Group by store to compare predictions and historical sales
historical_sales_by_store = historical_sales.groupby('Store')['Sales'].mean().reset_index()
predictions_by_store = output_df.groupby('Id')['Sales'].mean().reset_index()

# Plot comparison of sales predictions vs historical sales by store
plt.figure(figsize=(10, 6))
sns.barplot(x='Store', y='Sales', data=historical_sales_by_store, color='green', label='Historical Sales')
sns.barplot(x='Id', y='Sales', data=predictions_by_store, color='blue', alpha=0.5, label='Predictions')
plt.title('Sales Predictions vs Historical Sales (by store)')
plt.xlabel('Store')
plt.ylabel('Sales')
plt.legend()
plt.savefig('./images/sales_vs_historical.png')  # Save the image
plt.show()

# Group by date to compare sales predictions and historical sales by week
historical_sales_by_week = historical_sales.resample('W-Mon', on='Date')['Sales'].mean().reset_index()
predictions_by_week = test_df_cleaned.resample('W-Mon', on='Date')['Sales'].mean().reset_index()

# Plot comparison of sales predictions vs historical sales by week
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Sales', data=historical_sales_by_week, marker='o', color='green', label='Historical Sales')
sns.lineplot(x='Date', y='Sales', data=predictions_by_week, marker='o', color='blue', label='Predictions')
plt.title('Sales Predictions vs Historical Sales (by week)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.legend()
plt.savefig('./images/sales_vs_week.png')  # Save the image
plt.show()
