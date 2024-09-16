import requests
from bs4 import BeautifulSoup
import pandas as pd

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

# Função para pegar os reviews de uma página de um episódio
def get_imdb_reviews(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Encontra as seções de comentários
    review_containers = soup.find_all('div', class_='text show-more__control')
    
    reviews = [review.text.strip() for review in review_containers]
    
    return reviews

# Coletar todas as reviews de todos os episódios
all_reviews = []
for idx, url in enumerate(urls):
    reviews = get_imdb_reviews(url)
    episode = f'Episode {idx + 1}'
    for review in reviews:
        all_reviews.append([episode, review])

# Salvar em CSV
df_reviews = pd.DataFrame(all_reviews, columns=["Episode", "Review"])
df_reviews.to_csv('imdb_reviews_all_episodes.csv', index=False)