import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Download necessary NLTK data assets
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('omw-1.4', quiet=True)

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# -------------------------------------------------------------
# 1. Load Raw Data & Label Sentiments
# -------------------------------------------------------------
df_reviews = pd.read_csv("data/raw/reviews.csv")

# Map ratings: 1-2 -> Negative (0), 3 -> Neutral (1), 4-5 -> Positive (2)
def map_sentiment(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df_reviews['sentiment_target'] = df_reviews['rating_score'].apply(map_sentiment)

# -------------------------------------------------------------
# 2. Text Preprocessing Pipeline
# -------------------------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercasing & Punctuation Removal
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization, Stop-word removal, and Lemmatization
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

df_reviews['cleaned_review'] = df_reviews['review_text'].apply(preprocess_text)

# -------------------------------------------------------------
# 3. TF-IDF Feature Extraction
# -------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df_reviews['cleaned_review']).toarray()
y = df_reviews['sentiment_target'].values

# Split Train/Test Sets (80/20 stratified split)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------------
# 4. Sentiment Modeling (Multinomial Naive Bayes)
# -------------------------------------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Metrics Evaluation
acc = accuracy_score(y_test, y_pred_nb)
prec = precision_score(y_test, y_pred_nb, average='weighted')
rec = recall_score(y_test, y_pred_nb, average='weighted')
f1 = f1_score(y_test, y_pred_nb, average='weighted')

# Save Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Naive Bayes Sentiment Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('reports/figures/sentiment_confusion_matrix.png')
plt.close()

# -------------------------------------------------------------
# 5. Rule-Based VADER Comparison
# -------------------------------------------------------------
sia = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound <= -0.05:
        return 0  # Negative
    elif compound >= 0.05:
        return 2  # Positive
    else:
        return 1  # Neutral

df_reviews['vader_pred'] = df_reviews['review_text'].apply(get_vader_sentiment)
vader_acc = accuracy_score(df_reviews['sentiment_target'], df_reviews['vader_pred'])

# Save processed reviews dataset
df_reviews.to_csv("data/processed/reviews_sentiment_processed.csv", index=False)

# Output summary report
print("SUCCESS: Day 3 Sentiment Analysis & Preprocessing Completed.")
print(f"Naive Bayes Model Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
print(f"VADER Zero-Shot Model Accuracy: {vader_acc:.4f}")