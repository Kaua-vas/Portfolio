import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import matplotlib.pyplot as plt

# Paths to the data files (relative paths)
train_file_path = './train.csv'
store_file_path = './store.csv'

# Load the training data and store information
train_df = pd.read_csv(train_file_path)
store_df = pd.read_csv(store_file_path)

# Merge the training data with the store data on the "Store" column
df = pd.merge(train_df, store_df, on='Store')

# Handle missing values in the 'PromoInterval' column, fill missing with 'None'
df['PromoInterval'].fillna('None', inplace=True)
# Fill other missing values with 0
df.fillna(0, inplace=True)

# Preprocessing the 'Date' column, extract year, month, day, etc.
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
df['DayOfYear'] = df['Date'].dt.dayofyear

# Convert categorical variables to dummies (one-hot encoding) for model compatibility
df = pd.get_dummies(df, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'], drop_first=True)

# Remove rows where the store was closed (Open = 0)
df = df[df['Open'] == 1]

# Prepare input features (X) and target variable (y)
X = df.drop(columns=['Sales', 'Customers', 'Date', 'Open'])  # Features
y = df['Sales']  # Target variable (sales)

# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (standardize features by removing the mean and scaling to unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save the scaler to use it later when predicting on test data (relative path)
joblib.dump(scaler, './scaler_previsao_vendas.save')

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1]))  # First layer with 128 units
model.add(LeakyReLU(alpha=0.1))  # Activation function
model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(Dense(64))  # Second layer with 64 units
model.add(LeakyReLU(alpha=0.1))  # Activation function
model.add(Dropout(0.2))
model.add(Dense(32))  # Third layer with 32 units
model.add(LeakyReLU(alpha=0.1))  # Activation function
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer with 1 unit (sales prediction)

# Compile the model with Adam optimizer and mean squared error as the loss function
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Set callbacks: Early stopping to prevent overfitting, and checkpoint to save the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('./modelo_previsao_vendas_best.keras', save_best_only=True, monitor='val_loss')

# Train the model with training data, validating on the validation set
history = model.fit(X_train_scaled, y_train, 
                    validation_data=(X_val_scaled, y_val),
                    epochs=200, 
                    batch_size=32, 
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate the model on the validation set
val_mse, val_mae = model.evaluate(X_val_scaled, y_val)
print(f'Mean Squared Error on validation set: {val_mse}')
print(f'Mean Absolute Error on validation set: {val_mae}')

# Plot training and validation loss
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Loss (Training)')
plt.plot(history.history['val_loss'], label='Loss (Validation)')
plt.title('Loss During Training and Validation')
plt.xlabel('Epoch')
plt.ylabel('MSE (Mean Squared Error)')
plt.legend()
plt.grid(True)
plt.show()

# Save the final model after training (relative path)
model.save('./modelo_previsao_vendas_final.keras')

# Clear the TensorFlow session to free up resources
tf.keras.backend.clear_session()
