import pandas as pd
from transformers import pipeline

# Carregar o dataset de reviews processadas
df_reviews = pd.read_csv('imdb_processed_reviews_all_episodes.csv')

# Inicializar o pipeline de análise de sentimentos
sentiment_analysis = pipeline("sentiment-analysis")

# Função para análise de sentimentos
def analyze_sentiment(text):
    result = sentiment_analysis(text[:512])[0]  # Limitamos a 512 caracteres para o modelo
    return result['label'], result['score']

# Aplicar a análise de sentimentos em todas as reviews processadas
df_reviews['Sentiment'], df_reviews['Confidence'] = zip(*df_reviews['Processed_Review'].apply(analyze_sentiment))

# Salvar o arquivo final com os resultados da análise de sentimentos
df_reviews.to_csv('imdb_sentiment_reviews_all_episodes.csv', index=False)