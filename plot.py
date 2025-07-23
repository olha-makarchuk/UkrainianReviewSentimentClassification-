import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from bs4 import BeautifulSoup
import seaborn as sns

df = pd.read_csv('відгуки.csv')
df['label'] = df['label'].str.strip()

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

def plot_label_distribution(dataset):
    label_counts = dataset['label'].value_counts()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 8))
    light_colors = sns.husl_palette(n_colors=len(label_counts))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=light_colors)
    plt.title("Розподіл міток")
    plt.show()

def generate_wordcloud(text, title):
    all_text = " ".join(text)
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()


plot_label_distribution(df)

positive_text = df[df['label'] == 1]['cleaned_text'].tolist()
generate_wordcloud(positive_text, 'Positive Reviews')

negative_text = df[df['label'] == 0]['cleaned_text'].tolist()
generate_wordcloud(negative_text, 'Negative Reviews')

neutral_text = df[df['label'] == 2]['cleaned_text'].tolist()
generate_wordcloud(neutral_text, 'Neutral Reviews')