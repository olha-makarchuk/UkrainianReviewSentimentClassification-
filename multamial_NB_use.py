import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

loaded_model = joblib.load('naive_bayes_mModels/multamial_NB/naive_bayes_model.pkl')
loaded_vectorizer = joblib.load('Models/multamial_NB/vectorizer_NB.pkl')

data = pd.read_csv('Reviews/medium.csv')  
data['label'] = data['label'].str.strip()

new_reviews = data['text']
true_labels = data['label']

X_new = loaded_vectorizer.transform(new_reviews)

predictions = loaded_model.predict(X_new)

label_mapping = {'негативний': 0, 'нейтральний': 1, 'позитивний': 2}  

true_labels = true_labels.map(label_mapping)

accuracy = accuracy_score(true_labels, predictions)
print("Точність моделі:", accuracy)

cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['негативний', 'нейтральний', 'позитивний'],
            yticklabels=['негативний', 'нейтральний', 'позитивний'])
plt.title('MNB')
plt.show()