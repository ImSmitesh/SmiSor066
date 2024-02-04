# Using saved model to predict the given text

import joblib

vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('best_svm_model.pkl')
encoder = joblib.load("label_encoder.pkl")

def predict(text):
    """
    Function to predict the label of the input text using a trained model.
    
    Parameters:
    text (str): The input text to be predicted.
    
    Returns:
    str: The predicted label for the input text.
    """
    new_text_features = vectorizer.transform([text])
    y_pred = model.predict(new_text_features)
    y_pred_text = encoder.inverse_transform(y_pred)
    return y_pred_text[0]