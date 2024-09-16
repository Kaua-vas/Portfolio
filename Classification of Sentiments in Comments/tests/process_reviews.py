import pandas as pd
import re
import spacy

# Carregar o modelo de linguagem do SpaCy
nlp = spacy.load('en_core_web_sm')

# Função de pré-processamento usando SpaCy
def preprocess_text(text):
    if isinstance(text, str):  # Verificar se o texto é uma string válida
        # Processar o texto com SpaCy
        doc = nlp(text)
        # Lematizar, remover stopwords e pontuações
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)
    return ''

# Carregar o dataset de reviews
df_reviews = pd.read_csv('imdb_reviews_all_episodes.csv')

# Aplicar o pré-processamento a todas as reviews
df_reviews['Processed_Review'] = df_reviews['Review'].apply(preprocess_text)

# Salvar em um novo arquivo CSV
df_reviews.to_csv('imdb_processed_reviews_all_episodes.csv', index=False)