import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

# Load your trained model
model = load_model("sentiment.h5")  # Replace 'your_model.h5' with the actual path to your trained model file

# Load the CountVectorizer's vocabulary using pickle
with open('vectorizer_vocabulary.pkl', 'rb') as vocab_file:
    vocabulary = pickle.load(vocab_file)

# Load the CountVectorizer with the correct vocabulary
vectorizer = CountVectorizer(decode_error="replace", vocabulary=vocabulary)

# Define sentiment labels
sentiment_labels = ["Negative", "Neutral", "Positive"]

# Create a Streamlit app
st.title("Sentiment Analysis App")

# Create a text input box for user input
user_input = st.text_area("Enter a YouTube comment:")

# Create a button to trigger sentiment prediction
predict_button = st.button("Predict")

if predict_button:
    if user_input.strip():  # Check if user input is not empty or just whitespace
        # Tokenize and vectorize the user input
        user_input_vectorized = vectorizer.transform([user_input]).toarray()

        # Make predictions with your model
        predictions = model.predict(user_input_vectorized)

        # Interpret predictions
        predicted_label = sentiment_labels[np.argmax(predictions)]

        # Display the result
        st.write(f"Predicted Sentiment: {predicted_label}")
    else:
        st.warning("Please enter a valid YouTube comment.")
