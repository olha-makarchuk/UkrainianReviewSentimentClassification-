import tensorflow as tf
import re
from bs4 import BeautifulSoup
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

model = TFRobertaForSequenceClassification.from_pretrained('Models/roBerta_ukr/my_model_ukr')
tokenizer = RobertaTokenizer.from_pretrained('Models/roBerta_ukr/my_tokenizer_ukr')

def analyze_review(review_text):
    def clean_text(text):
        soup = BeautifulSoup(text, "html.parser")
        text = re.sub(r'\[[^]]*\]', '', soup.get_text())
        text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)
        return text

    cleaned_text = clean_text(review_text)
    
    encoding = tokenizer([cleaned_text], padding=True, truncation=True, return_tensors='tf', max_length=512)
    
    predictions = model.predict(dict(encoding))
    predicted_class = tf.argmax(predictions.logits, axis=-1).numpy()[0]
    
    label_map_inverse = {0: 'негативний', 1: 'позитивний', 2: 'нейтральний'}
    return label_map_inverse[predicted_class]

review_text = "Це був чудовий досвід, все сподобалося!"
predicted_category = analyze_review(review_text)
print(f"Відгук: {review_text}")
print(f"Передбачувана категорія: {predicted_category}")