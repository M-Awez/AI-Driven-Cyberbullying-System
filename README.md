# AI Cyberbullying Detector

A modern, AI-driven web application to detect cyberbullying in text using a pre-trained machine learning model.

## Features
- Detects cyberbullying in messages using an ML model (Logistic Regression/Naive Bayes)
- Clean, modern UI with Tailwind CSS
- Batch CSV upload for multiple messages (optional)
- Error handling and offensive word highlighting (optional)

## Project Structure
```
/AI-Driven Cyberbullying Detection
│
├── app.py                # Flask backend
├── model.py              # Model training and saving
├── requirements.txt      # Python dependencies
├── README.md             # Project overview
├── cyberbullying_model.pkl # Trained model
│
├── /static               # Static files (CSS, images)
│   └── ...
├── /templates            # HTML templates
│   └── ...
```

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python model.py`
3. Run the app: `python app.py`

## Credits
- Built with Flask, scikit-learn, Tailwind CSS
- Dataset: [Kaggle Cyberbullying Dataset](https://www.kaggle.com/datasets)
