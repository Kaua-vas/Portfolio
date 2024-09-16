import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Caminho para o arquivo de teste original
test_file_path = 'C:/Users/KELLY/Desktop/Portfolio/Inventory Demand Forecast/test.csv'

# Carregar o conjunto de teste
test_df = pd.read_csv(test_file_path)

# Preencher valores ausentes na coluna 'Open' com 1 (assumindo que as lojas estavam abertas)
test_df['Open'].fillna(1, inplace=True)

# Salvar o arquivo de teste atualizado
test_cleaned_file_path = 'C:/Users/KELLY/Desktop/Portfolio/Inventory Demand Forecast/test_cleaned.csv'
test_df.to_csv(test_cleaned_file_path, index=False)

print("Arquivo test_cleaned.csv gerado com sucesso!")

# Carregar o store.csv para fazer o merge com as informações de loja
store_file_path = 'C:/Users/KELLY/Desktop/Portfolio/Inventory Demand Forecast/store.csv'
store_df = pd.read_csv(store_file_path)

# Mesclar o conjunto de teste limpo com as informações de loja
test_df_cleaned = pd.read_csv(test_cleaned_file_path)
test_df_merged = pd.merge(test_df_cleaned, store_df, on='Store', how='left')

# Gerar dummies para as variáveis categóricas no conjunto de teste
categorical_columns = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
test_df_merged = pd.get_dummies(test_df_merged, columns=categorical_columns, drop_first=True)

# Carregar o scaler salvo para aplicar a normalização
scaler_path = 'C:/Users/KELLY/Desktop/Portfolio/scaler_previsao_vendas.save'
scaler = joblib.load(scaler_path)

# Garantir que as colunas do teste sejam as mesmas do treino (com ajuste para colunas vistas pelo scaler)
scaler_columns = scaler.feature_names_in_

# Adicionar colunas faltantes no conjunto de teste e preencher com 0
for col in scaler_columns:
    if col not in test_df_merged.columns:
        test_df_merged[col] = 0

# Remover colunas extras no conjunto de teste
test_df_cleaned_final = test_df_merged[scaler_columns]

# Aplicar a normalização nos dados de teste
X_test_scaled = scaler.transform(test_df_cleaned_final)

# Carregar o modelo treinado
model_path = 'C:/Users/KELLY/Desktop/Portfolio/modelo_previsao_vendas_final.keras'
modelo_salvo = load_model(model_path)

# Fazer previsões
previsoes = modelo_salvo.predict(X_test_scaled)

# Salvar as previsões em um novo arquivo CSV para submissão
output_file_path = 'C:/Users/KELLY/Desktop/Portfolio/Inventory Demand Forecast/submission_updated.csv'
output_df = pd.DataFrame({'Id': test_df_cleaned['Id'], 'Sales': previsoes.flatten()})

# Substituir valores NaN por 0 nas previsões
output_df['Sales'].fillna(0, inplace=True)

# Salvar o arquivo de submissão atualizado novamente
output_file_path_corrected = 'C:/Users/KELLY/Desktop/Portfolio/Inventory Demand Forecast/submission_corrected.csv'
output_df.to_csv(output_file_path_corrected, index=False)

print(f"Previsões salvas com correções em {output_file_path_corrected}")
