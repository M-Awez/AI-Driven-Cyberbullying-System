"""
Model training and saving script for AI Cyberbullying Detector
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import os

# Download NLTK stopwords if not already present
nltk.download('stopwords')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Load dataset (replace with actual path or download from Kaggle)
DATA_PATH = 'cyberbullying_tweets.csv'  # User must provide this file
if not os.path.exists(DATA_PATH):
    print(f"Dataset not found at {DATA_PATH}. Please download and place it in the project root.")
    exit(1)

df = pd.read_csv(DATA_PATH)
# Assume dataset has columns: 'tweet_text' and 'cyberbullying_type' or 'label'
if 'tweet_text' in df.columns:
    df['text'] = df['tweet_text']
if 'cyberbullying_type' in df.columns:
    # Save both binary and multiclass labels
    # Binary: 1 = any bullying, 0 = not_cyberbullying
    # Multiclass: keep original type
    df['label'] = (df['cyberbullying_type'] != 'not_cyberbullying').astype(int)
    df['bullying_type'] = df['cyberbullying_type']
elif 'label' not in df.columns:
    raise ValueError('Dataset must have a label or cyberbullying_type column.')

# Preprocess text
print('Cleaning text...')
df['clean_text'] = df['text'].astype(str).apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=200, class_weight='balanced', solver='liblinear')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print('Classification Report (Binary):')
print(classification_report(y_test, y_pred, target_names=['Not Cyberbullying', 'Cyberbullying']))

# Optional: Show confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Cyberbullying', 'Cyberbullying'], yticklabels=['Not Cyberbullying', 'Cyberbullying'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Offensive words (simple list for demo)
offensive_words = ['idiot', 'hate', 'stupid', 'dumb', 'loser', 'kill', 'ugly']

# Save model and vectorizer
joblib.dump({'model': model, 'vectorizer': vectorizer, 'offensive_words': offensive_words}, 'cyberbullying_model.pkl')
print('Model and vectorizer saved to cyberbullying_model.pkl')
