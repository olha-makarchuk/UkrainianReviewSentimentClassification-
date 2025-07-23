import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import seaborn as sns

model = TFBertForSequenceClassification.from_pretrained('Models/bert_en/my_model_ang')
tokenizer = BertTokenizer.from_pretrained('Models/bert_en/my_tokenizer_ang')

class_map = {0: "негативний", 1: "позитивний", 2: "нейтральний"}

def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='tf', max_length=512)
    
    predictions = model(inputs)
    predicted_class = tf.argmax(predictions.logits, axis=-1)
    
    return class_map[predicted_class.numpy()[0]]

df = pd.read_csv('Reviews/medium.csv')
df['label'] = df['label'].str.strip()

label_map = {"позитивний": 1, "нейтральний": 2, "негативний": 0}

y_true = df['label'].map(label_map).values 
y_pred = [predict_sentiment(text) for text in df['text']]  
y_pred_numeric = [label_map[label] for label in y_pred]

accuracy = accuracy_score(y_true, y_pred_numeric)
print("Точність моделі:", accuracy)

cm = confusion_matrix(y_true, y_pred_numeric, labels=[0, 2, 1])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['негативний', 'нейтральний', 'позитивний'],
            yticklabels=['негативний', 'нейтральний', 'позитивний'])
plt.title('BERT (англійська)')
plt.show()