import pandas as pd

# Caminho para o arquivo de teste original
test_file_path = 'C:/Users/KELLY/Desktop/Portfolio/Inventory Demand Forecast/test.csv'

# Carregar o conjunto de teste
test_df = pd.read_csv(test_file_path)

# Preencher valores ausentes na coluna 'Open' com 1 (assumindo que as lojas estavam abertas)
test_df['Open'].fillna(1, inplace=True)

# Salvar o arquivo atualizado
test_df.to_csv('C:/Users/KELLY/Desktop/Portfolio/Inventory Demand Forecast/test_cleaned.csv', index=False)

print("Arquivo test_cleaned.csv gerado com sucesso!")
