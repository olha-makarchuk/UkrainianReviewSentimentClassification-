import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from bs4 import BeautifulSoup

df = pd.read_csv('Reviews/train_reviews.csv')
df['label'] = df['label'].str.strip()

label_map = {'позитивний': 1, 'негативний': 0, 'нейтральний': 2}
df['label'] = df['label'].map(label_map)

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

train_texts, test_texts, train_labels, test_labels = train_test_split(df['cleaned_text'], 
                                                                      df['label'], test_size=0.2, 
                                                                      stratify=df['label'])

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, 
                                                                    train_labels, 
                                                                    test_size=0.1, 
                                                                    stratify=train_labels)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_sentences(sentences):
    return tokenizer(sentences, padding=True, truncation=True, return_tensors='tf', max_length=512)

train_encodings = encode_sentences(train_texts.tolist())
val_encodings = encode_sentences(val_texts.tolist())
test_encodings = encode_sentences(test_texts.tolist())

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels.tolist()))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels.tolist()))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels.tolist()))

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset.batch(8), validation_data=val_dataset.batch(8), epochs=3)

model.save_pretrained('my_model_ang')
tokenizer.save_pretrained('my_tokenizer_ang')

test_loss, test_accuracy = model.evaluate(test_dataset.batch(8))
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

predictions = model.predict(test_dataset.batch(8))
predicted_classes = tf.argmax(predictions.logits, axis=-1)

print(classification_report(test_labels, predicted_classes))