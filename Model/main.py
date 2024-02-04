# Hosting model using FastAPI.

from fastapi import FastAPI
from pydantic import BaseModel
from app import *

model_version = '1.4.0'
model_name = 'SVM'

app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    prediction: str

@app.get("/")
def home():
    """
    A function that handles the home endpoint, returning health check status, model version, and model name.
    """
    return {"health_check": "OK", "model_version": model_version, "model_name": model_name}

@app.post("/predict/", response_model=PredictionOut)
def predict_text(text_in: TextIn):
    """
    Predicts the label for the given input text using the provided vectorizer and model.

    Parameters:
    - text_in: TextIn, input text to be predicted

    Returns:
    - dict, containing the predicted label
    """
    new_text_features = vectorizer.transform([text_in.text])
    y_pred = model.predict(new_text_features)
    prediction = encoder.inverse_transform(y_pred)[0]
    return {"prediction": prediction}