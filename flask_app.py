from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model("sentimentt.h5")  # Replace 'your_model.h5' with the actual path to your trained model file

# Load the CountVectorizer's vocabulary using pickle
vectorizer = CountVectorizer(decode_error="replace")
with open('vectorizer_vocabulary.pkl', 'rb') as vocab_file:
    vectorizer.vocabulary_ = np.load(vocab_file, allow_pickle=True)

# Define sentiment labels
sentiment_labels = ["Negative", "Neutral", "Positive"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.form.get('comments')

    # Tokenize and vectorize the input comments
    new_comments_vectorized = vectorizer.transform([data]).toarray()

    # Make predictions with your model
    predictions = model.predict(new_comments_vectorized)

    # Interpret predictions
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_sentiment = sentiment_labels[predicted_labels[0]]

    return render_template('result.html', sentiment=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
