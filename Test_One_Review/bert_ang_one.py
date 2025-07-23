import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import re
from bs4 import BeautifulSoup

model = TFBertForSequenceClassification.from_pretrained('Models/bert_ang/my_model_ang')
tokenizer = BertTokenizer.from_pretrained('Models/bert_ang/my_tokenizer_ang')

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)
    return text

def analyze_review(review_text):
    cleaned_review = clean_text(review_text)
    
    encoding = tokenizer(cleaned_review, padding=True, truncation=True, return_tensors='tf', max_length=512)
    
    prediction = model.predict(dict(encoding))
    predicted_class = tf.argmax(prediction.logits, axis=-1).numpy()[0]
    
    label_map = {0: 'негативний', 1: 'позитивний', 2: 'нейтральний'}
    return label_map[predicted_class]

review = "Це був чудовий досвід, все сподобалося!"
result = analyze_review(review)
print(f"Результат аналізу: {result}")