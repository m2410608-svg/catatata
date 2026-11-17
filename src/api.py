from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import yaml
import json
from sklearn.linear_model import LogisticRegression

app = FastAPI(
    title="Images Classifier API",
    description="API для классификации изображений одежды",
    version="1.0.0"
)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()


model_path = os.path.join(os.path.dirname(__file__), "..", config['paths']['model'])
with open(model_path, "r") as f:
        model_data = json.load(f)
 
model = LogisticRegression()
model.coef_ = np.array(model_data["coef"])
model.intercept_ = np.array(model_data["intercept"])
model.classes_ = np.array(model_data["classes"])

pipeline = model


class ImageInput(BaseModel):
    data: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [0.0] * 784
            }
        }

class BatchInput(BaseModel):
    data: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float

class BatchResponse(BaseModel):
    predictions: List[int]

@app.get("/")
def root():
    return {
        "message": config['api']['title'],
        "version": config['api']['version'],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "loaded",
        "model_path": str(model_path),
        "task": config['data_task_name']['name']
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(image: ImageInput):
    try:
        n_features = config['data']['n_features']
        if len(image.data) != n_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {n_features} features (28x28), got {len(image.data)}"
            )
        
        X = np.array(image.data).reshape(1, -1)
        prediction = pipeline.predict(X)[0]
        probabilities = pipeline.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        return {
            "prediction": int(prediction),
            "confidence": round(confidence, 4)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(images: BatchInput):
    try:
        if not images.data:
            raise HTTPException(status_code=400, detail="Empty batch")
        
        n_features = config['data']['n_features']
        
        for i, img in enumerate(images.data):
            if len(img) != n_features:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Image {i}: expected {n_features} features, got {len(img)}"
                )
        
        X = np.array(images.data)
        predictions = pipeline.predict(X)
        
        return {
            "predictions": [int(p) for p in predictions]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")