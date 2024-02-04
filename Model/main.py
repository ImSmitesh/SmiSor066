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
    return {"health_check": "OK", "model_version": model_version, "model_name": model_name}

@app.post("/predict/", response_model=PredictionOut)
async def predict_text(text_in: TextIn):
    new_text_features = vectorizer.transform([text_in.text])
    y_pred = model.predict(new_text_features)
    prediction = encoder.inverse_transform(y_pred)[0]
    return {"prediction": prediction}