import re
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())  
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)  
    return text

def load_model_and_vectorizer():
    mnb = joblib.load('Models/multamial_NB/naive_bayes_model.pkl')
    vectorizer = joblib.load('Models/multamial_NB/vectorizer_NB.pkl')
    return mnb, vectorizer

def analyze_review(review_text):
    cleaned_review = clean_text(review_text)
    
    mnb, vectorizer = load_model_and_vectorizer()
    
    review_vec = vectorizer.transform([cleaned_review])
    
    prediction = mnb.predict(review_vec)
    label_map = {0: 'Негативний', 1: 'Позитивний', 2: 'Нейтральний'}
    predicted_label = label_map[prediction[0]]
    
    return predicted_label

review = "Це був чудовий досвід, все сподобалося!"
result = analyze_review(review)
print(f"Результат аналізу: {result}")
