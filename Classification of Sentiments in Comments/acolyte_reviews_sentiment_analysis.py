import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from transformers import pipeline

# URLs dos episódios
urls = [
    'https://www.imdb.com/title/tt13820376/reviews/?ref_=tt_urv',  # Episode 1
    'https://www.imdb.com/title/tt15387022/reviews/?ref_=tt_urv',  # Episode 2
    'https://www.imdb.com/title/tt15387026/reviews/?ref_=tt_urv',  # Episode 3
    'https://www.imdb.com/title/tt15387030/reviews/?ref_=tt_urv',  # Episode 4
    'https://www.imdb.com/title/tt15387052/reviews/?ref_=tt_urv',  # Episode 5
    'https://www.imdb.com/title/tt15387054/reviews/?ref_=tt_urv',  # Episode 6
    'https://www.imdb.com/title/tt15387058/reviews/?ref_=tt_urv',  # Episode 7
    'https://www.imdb.com/title/tt16274268/reviews/?ref_=tt_urv'   # Episode 8
]

# Função de scraping das reviews de cada página do episódio
def get_imdb_reviews(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Encontra as seções de comentários
    review_containers = soup.find_all('div', class_='text show-more__control')
    
    reviews = [review.text.strip() for review in review_containers]
    
    return reviews

# Função de pré-processamento usando SpaCy
nlp = spacy.load('en_core_web_sm')
def preprocess_text(text):
    if isinstance(text, str):
        doc = nlp(text)
        # Lematizar, remover stopwords e pontuações
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)
    return ''

# Inicializar o pipeline de análise de sentimentos da Hugging Face
sentiment_analysis = pipeline("sentiment-analysis")

# Função para análise de sentimentos
def analyze_sentiment(text):
    result = sentiment_analysis(text[:512])[0]  # Limitando a 512 caracteres para o modelo
    return result['label'], result['score']

# Processo completo: Scraping, Pré-processamento e Análise de Sentimentos
all_reviews = []
for idx, url in enumerate(urls):
    reviews = get_imdb_reviews(url)  # Coletar reviews do episódio
    episode = f'Episode {idx + 1}'
    
    for review in reviews:
        processed_review = preprocess_text(review)  # Pré-processar review
        sentiment, confidence = analyze_sentiment(processed_review)  # Análise de sentimentos
        all_reviews.append([episode, review, processed_review, sentiment, confidence])

# Criar DataFrame e salvar resultados
df = pd.DataFrame(all_reviews, columns=['Episode', 'Review', 'Processed_Review', 'Sentiment', 'Confidence'])
df.to_csv('imdb_sentiment_reviews_all_episodes.csv', index=False)

print("Análise de sentimentos concluída e salva em 'imdb_sentiment_reviews_all_episodes.csv'")