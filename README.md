# UkrainianReviewSentimentClassification

This project aims to classify Ukrainian-language user reviews into three sentiment categories — positive, neutral, and negative — using multiple machine learning and deep learning models trained on the same dataset. The objective is to compare the performance of traditional and transformer-based approaches on this task.

---

## 📄 Full Analysis Report
A comprehensive PDF report is included, which covers:

 - 🔍 Dataset and preprocessing overview

 - 🤖 Model selection rationale

 - 📊 Detailed evaluation metrics & visualizations

 - 💡 Comparative performance analysis

 - ⚠️ Challenges specific to Ukrainian NLP

 - ✅ Final conclusions and recommendations

📥 [View Full Report (PDF)](https://github.com/olha-makarchuk/UkrainianReviewSentimentClassification/blob/Main/sentiment-classification-ukrainian-reviews-report.pdf)


| Model                       | Type             | Description                                                                              |
| --------------------------- | ---------------- | ---------------------------------------------------------------------------------------- |
| **BERT (Multilingual)**     | Transformer (DL) | Fine-tuned version of `bert-base-uncased`, trained for 3-class sentiment classification. |
| **RoBERTa (YouScan)**       | Transformer (DL) | Pre-trained `youscan/ukr-roberta-base`, fine-tuned on Ukrainian review data.             |
| **Multinomial Naive Bayes** | Classical ML     | Uses `CountVectorizer` and `MultinomialNB` from `scikit-learn` as a baseline classifier. |

---

## 🗂 Dataset
The dataset contains Ukrainian-language user reviews labeled as:

0 — Negative

1 — Positive

2 — Neutral

Located under the Reviews/ directory:

| File                | Purpose                                   |
| ------------------- | ----------------------------------------- |
| `train_reviews.csv` | Full dataset for training and evaluation  |
| `small.csv`         | Small subset for rapid testing            |
| `medium.csv`        | Medium-sized dataset for stability checks |
| `big.csv`           | Full-size dataset for final benchmarks    |

---

## 🧪 Evaluation Metrics
Each model is evaluated using:

 - Accuracy

 - Precision / Recall / F1-score

 - Confusion Matrix

 - Log Loss (where applicable)

 - Visualization of training metrics and evaluation results is done via matplotlib and seaborn.

## 📁 Models & Files
Due to GitHub's file size restrictions, trained model and tokenizer/vectorizer files are not stored directly in this repository.

📦 Download Pretrained Models:
You can download all pretrained models, tokenizers, and vectorizers from Google Drive — see the README.txt or .txt files inside each model folder for detailed instructions and download links.

⚙️ Recreate Models Locally:
Alternatively, you can retrain all models from scratch using the provided code in the Create_models directory. This includes:

 - Fine-tuning the BERT and RoBERTa models

 - Training the Multinomial Naive Bayes model

 - Generating the corresponding Tokenizers and Vectorizers

This approach ensures full reproducibility of the project.
