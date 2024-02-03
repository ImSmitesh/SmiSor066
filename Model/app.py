from flask import Flask, request, jsonify
import joblib

# Load your trained TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load your trained model
model = joblib.load('SVM_model.pkl')

# Load your label encoder
encoder = joblib.load("l_encoder_vectorizer.pkl")

app = Flask(__name__)

def predict(text):
    # Transform the input text using the fitted vectorizer
    new_text_features = vectorizer.transform([text])

    # Make predictions
    y_pred = model.predict(new_text_features)

    # Convert numeric predictions to text labels
    y_pred_text = encoder.inverse_transform(y_pred)

    return y_pred_text[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input text from the user
        text = request.form['text']
        prediction = predict(text)
        return jsonify({'prediction': prediction})
    else:
        return '''
            <form method="post">
                <label for="text">Enter text:</label><br>
                <input type="text" id="text" name="text"><br>
                <input type="submit" value="Submit">
            </form>
        '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)