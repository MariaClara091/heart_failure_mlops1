import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

def basic_analysis():
    """Análisis básico sin evidently"""
    
    print("Análisis básico del dataset...")
    df = pd.read_csv("heart.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"HeartDisease: {df['HeartDisease'].value_counts().to_dict()}")
    try:
        model = joblib.load("app/model.joblib")
        print(f"Modelo: {type(model).__name__}")
        if hasattr(model, 'best_params_'):
            print(f"Mejores parámetros: {model.best_params_}")
    except Exception as e:
        print(f"Error: {e}")

    X = df.drop("HeartDisease", axis=1)
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = pd.factorize(X[col])[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, df["HeartDisease"], test_size=0.3, random_state=42)
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print("Estadísticas train vs test:")
    print(f"   Media Age - Train: {X_train['Age'].mean():.1f}, Test: {X_test['Age'].mean():.1f}")
    
    print("Análisis completado")

if __name__ == "__main__":
    basic_analysis()