import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv('imdb_sentiment_reviews_all_episodes.csv')

# Gráfico 1: Distribuição de sentimentos por episódio
plt.figure(figsize=(10, 6))
sns.countplot(x='Episode', hue='Sentiment', data=df)
plt.title('Distribuição de Sentimentos por Episódio')
plt.ylabel('Número de Reviews')
plt.xlabel('Episódio')
plt.legend(title='Sentimento')
plt.show()

# Gráfico 2: Média da confiança do modelo por episódio
plt.figure(figsize=(10, 6))
sns.barplot(x='Episode', y='Confidence', hue='Sentiment', data=df)
plt.title('Confiança Média do Modelo por Episódio e Sentimento')
plt.ylabel('Confiança Média')
plt.xlabel('Episódio')
plt.legend(title='Sentimento')
plt.show()

# Gráfico 3: Evolução dos sentimentos ao longo dos episódios
sentiment_counts = df.groupby(['Episode', 'Sentiment']).size().unstack().fillna(0)
sentiment_counts.plot(kind='line', figsize=(10, 6))
plt.title('Evolução dos Sentimentos ao Longo dos Episódios')
plt.ylabel('Número de Reviews')
plt.xlabel('Episódio')
plt.show()