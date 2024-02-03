import joblib

# Load your trained TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load your trained model
model = joblib.load('Gradient Boosting_model.pkl')

# Load your label encoder
encoder = joblib.load("l_encoder_vectorizer.pkl")

def predict(text):
    # Transform the input text using the fitted vectorizer
    new_text_features = vectorizer.transform([text])

    # Make predictions
    y_pred = model.predict(new_text_features)

    # Convert numeric predictions to text labels
    y_pred_text = encoder.inverse_transform(y_pred)

    return y_pred_text[0]

if __name__ == '__main__':
    # Get input text from the user
    text = input("Enter text: ")
    prediction = predict(text)
    print("Predicted class:", prediction)
