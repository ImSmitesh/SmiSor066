from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib

# Load your trained TF-IDF vectorizer
vectorizer = joblib.load(open('tfidf_vectorizer.pkl.', 'rb'))

# Load your trained model
model = pickle.load(open('Gradient Boosting_model.pkl', 'rb'))

# Get input text from the user
text = input("Enter text: ")

# Transform the input text using the fitted vectorizer
new_text_features = vectorizer.transform([text])

# Make predictions
print(model.predict(new_text_features))
