"""
Flask backend for AI Cyberbullying Detector
"""
import os
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd

# Load the trained model and vectorizer
MODEL_PATH = 'cyberbullying_model.pkl'
model_data = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flash messages

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_text = ''
    error = None
    offensive_words = []
    if request.method == 'POST':
        input_text = request.form.get('message', '').strip()
        if not input_text:
            error = 'Please enter a message.'
        elif not model_data:
            error = 'Model not found. Please train the model first.'
        else:
            vectorizer = model_data['vectorizer']
            model = model_data['model']
            features = vectorizer.transform([input_text])
            pred = model.predict(features)[0]
            prediction = 'Cyberbullying' if pred == 1 else 'Not Cyberbullying'
            # Offensive word highlight (optional, simple demo)
            if 'offensive_words' in model_data:
                offensive_words = [w for w in model_data['offensive_words'] if w in input_text.lower()]
    return render_template('index.html', prediction=prediction, input_text=input_text, error=error, offensive_words=offensive_words)

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    results = None
    error = None
    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file part.'
        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No selected file.'
            elif not model_data:
                error = 'Model not found. Please train the model first.'
            else:
                try:
                    df = pd.read_csv(file)
                    if 'message' not in df.columns:
                        error = 'CSV must have a "message" column.'
                    else:
                        vectorizer = model_data['vectorizer']
                        model = model_data['model']
                        features = vectorizer.transform(df['message'].astype(str))
                        preds = model.predict(features)
                        df['Prediction'] = ['Cyberbullying' if p == 1 else 'Not Cyberbullying' for p in preds]
                        results = df[['message', 'Prediction']].to_dict('records')
                except Exception as e:
                    error = f'Error processing file: {e}'
    return render_template('batch.html', results=results, error=error)

@app.route('/batch_text', methods=['GET', 'POST'])
def batch_text():
    results = None
    error = None
    input_text = ''
    if request.method == 'POST':
        input_text = request.form.get('messages', '').strip()
        if not input_text:
            error = 'Please enter at least one message.'
        elif not model_data:
            error = 'Model not found. Please train the model first.'
        else:
            lines = [line.strip() for line in input_text.split('\n') if line.strip()]
            vectorizer = model_data['vectorizer']
            model = model_data['model']
            features = vectorizer.transform(lines)
            preds = model.predict(features)
            results = [
                {'message': msg, 'Prediction': 'Cyberbullying' if p == 1 else 'Not Cyberbullying'}
                for msg, p in zip(lines, preds)
            ]
    return render_template('batch_text.html', results=results, error=error, input_text=input_text)


if __name__ == '__main__':
    app.run(debug=True)
