import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
import re
from bs4 import BeautifulSoup

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].str.strip()
    return df

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)
    return text

def preprocess_dataset(dataset):
    train_text = dataset['text'].values
    train_labels = LabelEncoder().fit_transform(dataset['label'])
    
    validation_text = dataset.sample(frac=0.1, random_state=42)['text'].values
    validation_labels = LabelEncoder().fit_transform(dataset.sample(frac=0.1, random_state=42)['label'])
    test_text = dataset.sample(frac=0.1, random_state=24)['text'].values
    test_labels = LabelEncoder().fit_transform(dataset.sample(frac=0.1, random_state=24)['label'])

    return train_text, train_labels, validation_text, validation_labels, test_text, test_labels

def vectorize_data(tr_text, ts_text):
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(tr_text)
    X_test_vec = vectorizer.transform(ts_text)
    
    return X_train_vec, X_test_vec, vectorizer

def train_naive_bayes(X_train_vec, y_train):
    mnb = MultinomialNB(alpha=0.1, fit_prior=True, force_alpha=True)
    mnb.fit(X_train_vec, y_train)
    
    return mnb

def evaluate_model(mnb, X_test_vec, y_test):
    y_pred_mnb = mnb.predict(X_test_vec)
    y_prob_mnb = mnb.predict_proba(X_test_vec) 
    
    accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
    precision_mnb = precision_score(y_test, y_pred_mnb, average='weighted', zero_division=0)
    recall_mnb = recall_score(y_test, y_pred_mnb, average='weighted', zero_division=0)
    f1_mnb = f1_score(y_test, y_pred_mnb, average='weighted')
    loss_mnb = log_loss(y_test, y_prob_mnb)

    print(f"Accuracy: {accuracy_mnb:.2f}")
    print(f"Precision: {precision_mnb:.2f}")
    print(f"Recall: {recall_mnb:.2f}")
    print(f"F1-score: {f1_mnb:.2f}")
    print(f"Loss: {loss_mnb:.2f}")

def plot_confusion_matrix(model, X_test_vec, y_test, model_name="Model"):
    X_test_dense = X_test_vec.toarray() if hasattr(X_test_vec, 'toarray') else X_test_vec
    y_pred = model.predict(X_test_dense)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Позитивний', 'Нейтральний', 'Негативний'], 
                yticklabels=['Позитивний', 'Нейтральний', 'Негативний'])
    plt.title(f'Матриця плутанини - {model_name}')
    plt.xlabel('Прогнозовані категорії')
    plt.ylabel('Справжні категорії')
    plt.show()

def main(file_path):
    dataset = load_dataset(file_path)
    dataset['cleaned_text'] = dataset['text'].apply(clean_text)
    
    train_text, train_labels, validation_text, validation_labels, test_text, test_labels = preprocess_dataset(dataset)

    X_train_vec, X_test_vec, vectorizer = vectorize_data(train_text, test_text)

    mnb = train_naive_bayes(X_train_vec, train_labels)
    
    evaluate_model(mnb, X_test_vec, test_labels)

    plot_confusion_matrix(mnb, X_test_vec, test_labels, "Multinomial Naive Bayes")
    
    joblib.dump(mnb, 'naive_bayes_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    file_path = "Reviews/train_reviews.csv"
    main(file_path)