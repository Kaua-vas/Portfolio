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

# Caminho para os arquivos
train_file_path = 'C:\\Users\\KELLY\\Desktop\\Portfolio\\Inventory Demand Forecast\\train.csv'
store_file_path = 'C:\\Users\\KELLY\\Desktop\\Portfolio\\Inventory Demand Forecast\\store.csv'

# Carregar os dados de treino e informações das lojas
train_df = pd.read_csv(train_file_path)
store_df = pd.read_csv(store_file_path)

# Mesclar os dados de treino com os dados das lojas
df = pd.merge(train_df, store_df, on='Store')

# Tratar valores ausentes
df['PromoInterval'].fillna('None', inplace=True)
df.fillna(0, inplace=True)  # Preencher outros valores ausentes com 0

# Pré-processamento dos dados
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
df['DayOfYear'] = df['Date'].dt.dayofyear

# Transformar variáveis categóricas em variáveis dummies
df = pd.get_dummies(df, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'], drop_first=True)

# Remover dados onde a loja estava fechada
df = df[df['Open'] == 1]

# Preparar os dados de entrada e saída
X = df.drop(columns=['Sales', 'Customers', 'Date', 'Open'])  # Features
y = df['Sales']  # Target

# Dividir os dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados e salvar o scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Salvar o scaler
joblib.dump(scaler, 'C:\\Users\\KELLY\\Desktop\\Portfolio\\scaler_previsao_vendas.save')

# Definir o modelo
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1]))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compilar o modelo com uma taxa de aprendizado menor
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks para EarlyStopping e salvar o melhor modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('C:\\Users\\KELLY\\Desktop\\Portfolio\\modelo_previsao_vendas_best.keras', save_best_only=True, monitor='val_loss')

# Treinar o modelo
history = model.fit(X_train_scaled, y_train, 
                    validation_data=(X_val_scaled, y_val),
                    epochs=200, 
                    batch_size=32, 
                    callbacks=[early_stopping, model_checkpoint])

# Avaliar o modelo no conjunto de validação
val_mse, val_mae = model.evaluate(X_val_scaled, y_val)
print(f'Mean Squared Error no conjunto de validação: {val_mse}')
print(f'Mean Absolute Error no conjunto de validação: {val_mae}')

# Plotar os gráficos de perda
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Loss (Treinamento)')
plt.plot(history.history['val_loss'], label='Loss (Validação)')
plt.title('Perda durante o Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('MSE (Mean Squared Error)')
plt.legend()
plt.grid(True)
plt.show()

# Salvar o modelo final após o treinamento
model.save('C:\\Users\\KELLY\\Desktop\\Portfolio\\modelo_previsao_vendas_final.keras')

# Limpar a sessão do TensorFlow
tf.keras.backend.clear_session()
