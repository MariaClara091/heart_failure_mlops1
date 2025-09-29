from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

# Cargar el modelo entrenado
model = joblib.load("app/model.joblib")
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Definir el esquema de datos de entrada
class PatientData(BaseModel):
    Age: int
    Sex: int
    ChestPainType: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: int 
    MaxHR: int
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int

class PredictionResponse(BaseModel):
    heart_disease_probability: float
    prediction: int
    risk_level: str
    status: str

@app.get("/")
def read_root():
    return {
        "message": "Heart Disease Prediction API", 
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "heart-disease-api"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    try:
        # Convertir datos a array numpy
        features = np.array([
            data.Age, data.Sex, data.ChestPainType, data.RestingBP, 
            data.Cholesterol, data.FastingBS, data.RestingECG, data.MaxHR,  # ❌ CORREGIDO
            data.ExerciseAngina, data.Oldpeak, data.ST_Slope
        ]).reshape(1, -1)
        
        # Predecir probabilidad
        probability = model.predict_proba(features)[0][1]
        
        # Determinar predicción y nivel de riesgo
        prediction = 1 if probability > 0.5 else 0
        
        if probability < 0.3:
            risk_level = "Bajo"
        elif probability < 0.7:
            risk_level = "Moderado"
        else:
            risk_level = "Alto"
        
        return {
            "heart_disease_probability": float(probability),
            "prediction": prediction,
            "risk_level": risk_level,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "heart_disease_probability": 0.0,
            "prediction": -1,
            "risk_level": "Error",
            "status": f"error: {str(e)}"
        }

# Endpoint para obtener información del modelo
@app.get("/model-info")
def model_info():
    try:
        best_params = model.best_params_ if hasattr(model, 'best_params_') else "No disponible"
        return {
            "model_type": type(model).__name__,
            "best_parameters": best_params,
            "features_used": 11
        }
    except:
        return {"model_type": "Modelo cargado", "features_used": 11}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)