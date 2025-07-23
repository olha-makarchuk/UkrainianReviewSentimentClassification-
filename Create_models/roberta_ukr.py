import pandas as pd
import tensorflow as tf
from transformers import  TFRobertaForSequenceClassification, RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from sklearn.metrics import classification_report, f1_score
import seaborn as sns

df = pd.read_csv('відгуки.csv')
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

tokenizer = RobertaTokenizer.from_pretrained("youscan/ukr-roberta-base")

def encode_sentences(sentences):
    return tokenizer(sentences, padding=True, truncation=True, return_tensors='tf', max_length=512)

train_encodings = encode_sentences(train_texts.tolist())
val_encodings = encode_sentences(val_texts.tolist())
test_encodings = encode_sentences(test_texts.tolist())

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels.tolist()))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels.tolist()))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels.tolist()))

model = TFRobertaForSequenceClassification.from_pretrained("youscan/ukr-roberta-base", 
                                                             num_labels=3, 
                                                             from_pt=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset.batch(8), validation_data=val_dataset.batch(8), epochs=3)

model.save_pretrained('my_model_ukr')
tokenizer.save_pretrained('my_tokenizer_ukr')

test_loss, test_accuracy = model.evaluate(test_dataset.batch(8))
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

predictions = model.predict(test_dataset.batch(8))
predicted_classes = tf.argmax(predictions.logits, axis=-1)

print(classification_report(test_labels, predicted_classes))

f1 = f1_score(test_labels, predicted_classes, average='weighted')
print(f"F1 Score: {f1}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()