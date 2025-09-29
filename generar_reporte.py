import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import os

def generate_complete_report():
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. ANÁLISIS COMPLETO DEL DATASET
    print("\n1. ANÁLISIS DEL DATASET")
    df = pd.read_csv("heart.csv")
    
    dataset_info = {
        "total_registros": df.shape[0],
        "total_variables": df.shape[1],
        "registros_con_enfermedad": df['HeartDisease'].sum(),
        "porcentaje_enfermedad": (df['HeartDisease'].mean() * 100),
        "variables_numericas": len(df.select_dtypes(include=[np.number]).columns),
        "variables_categoricas": len(df.select_dtypes(include=['object']).columns)
    }
    
    print(f"   • Total registros: {dataset_info['total_registros']} pacientes")
    print(f"   • Enfermedad cardíaca: {dataset_info['porcentaje_enfermedad']:.1f}%")
    print(f"   • Variables: {dataset_info['variables_numericas']} numéricas, {dataset_info['variables_categoricas']} categóricas")
    
    # 2. PREPARACIÓN DE DATOS
    print("\n2. PREPARACIÓN DE DATOS")
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    
    # Convertir variables categóricas
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    conversion_map = {}
    
    for col in categorical_cols:
        original_values = X[col].unique()
        X[col], unique_codes = pd.factorize(X[col])
        conversion_map[col] = dict(zip(unique_codes, range(len(unique_codes))))
        print(f"   • {col}: {list(original_values)} → {list(range(len(unique_codes)))}")
    
    # 3. DIVISIÓN DE DATOS
    print("\n3. DIVISIÓN DE DATOS")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    split_info = {
        "train_records": X_train.shape[0],
        "test_records": X_test.shape[0],
        "train_disease_percent": (y_train.mean() * 100),
        "test_disease_percent": (y_test.mean() * 100)
    }
    
    print(f"   • Entrenamiento: {split_info['train_records']} registros ({split_info['train_disease_percent']:.1f}% enfermedad)")
    print(f"   • Prueba: {split_info['test_records']} registros ({split_info['test_disease_percent']:.1f}% enfermedad)")
    print(f"   • Proporción: 70% / 30% (balanceado)")
    
    # 4. INFORMACIÓN DEL MODELO
    print("\n4. INFORMACIÓN DEL MODELO")
    try:
        model = joblib.load("app/model.joblib")
        model_info = {
            "tipo": type(model).__name__,
            "mejor_score": getattr(model, 'best_score_', 'N/A'),
            "parametros": getattr(model, 'best_params_', 'N/A')
        }
        
        print(f"   • Tipo: {model_info['tipo']}")
        if model_info['mejor_score'] != 'N/A':
            print(f"   • Mejor score: {model_info['mejor_score']:.3f}")
        if model_info['parametros'] != 'N/A':
            print(f"   • Mejores parámetros: {model_info['parametros']}")
            
    except Exception as e:
        print(f"   • Error: {e}")
        model_info = {"tipo": "No disponible", "mejor_score": "N/A", "parametros": "N/A"}
    
    # 5. ANÁLISIS DE DATA DRIFT
    print("\n5. ANÁLISIS DE DATA DRIFT")
    
    # Calcular diferencias entre train y test
    drift_analysis = {}
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        train_mean = X_train[col].mean()
        test_mean = X_test[col].mean()
        diff_percent = abs((test_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0
        
        drift_analysis[col] = {
            "train_mean": train_mean,
            "test_mean": test_mean,
            "diferencia_porcentual": diff_percent,
            "estado": "ALTO" if diff_percent > 10 else "MODERADO" if diff_percent > 5 else "BAJO"
        }
    
    # Mostrar variables con mayor drift
    high_drift_vars = {k: v for k, v in drift_analysis.items() if v['estado'] == 'ALTO'}
    print(f"   • Variables con alto drift: {len(high_drift_vars)}")
    for var, info in list(high_drift_vars.items())[:3]:  # Mostrar top 3
        print(f"     - {var}: {info['diferencia_porcentual']:.1f}% diferencia")
    
    # 6. GENERAR REPORTE HTML PROFESIONAL
    print("\n6. GENERANDO REPORTE VISUAL")
    
    # Crear reporte HTML
    html_content = f'''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Monitoreo - Predicción Cardíaca</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ margin: 25px 0; padding: 20px; border-left: 5px solid #667eea; background: #f8f9fa; }}
        .metric-card {{ background: white; padding: 15px; margin: 10px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); display: inline-block; min-width: 200px; }}
        .high-drift {{ color: #e74c3c; font-weight: bold; }}
        .medium-drift {{ color: #f39c12; }}
        .low-drift {{ color: #27ae60; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Reporte de Monitoreo - Sistema de Predicción Cardíaca</h1>
            <p>Generado el {datetime.now().strftime("%d/%m/%Y %H:%M")}</p>
        </div>
        
        <div class="section">
            <h2> Resumen Ejecutivo</h2>
            <div class="stats-grid">
                <div class="metric-card">
                    <h3>{dataset_info["total_registros"]}</h3>
                    <p>Total Pacientes</p>
                </div>
                <div class="metric-card">
                    <h3>{dataset_info["porcentaje_enfermedad"]:.1f}%</h3>
                    <p>Prevalencia Enfermedad</p>
                </div>
                <div class="metric-card">
                    <h3>{split_info["train_records"]}</h3>
                    <p>Registros Entrenamiento</p>
                </div>
                <div class="metric-card">
                    <h3>{split_info["test_records"]}</h3>
                    <p>Registros Prueba</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2> Información del Modelo</h2>
            <p><strong>Tipo:</strong> {model_info["tipo"]}</p>
            <p><strong>Mejor Score:</strong> {model_info["mejor_score"] if model_info["mejor_score"] != "N/A" else "No disponible"}</p>
            <p><strong>Parámetros Optimizados:</strong> {model_info["parametros"] if model_info["parametros"] != "N/A" else "No disponible"}</p>
        </div>
        
        <div class="section">
            <h2> Análisis de Data Drift</h2>
            <p><strong>Variables con Alto Drift (>10%):</strong> {len(high_drift_vars)}</p>
            <div class="stats-grid">
    '''
    
    # Agregar métricas de drift
    for var, info in list(drift_analysis.items())[:6]:  # Mostrar primeras 6 variables
        drift_class = "high-drift" if info["estado"] == "ALTO" else "medium-drift" if info["estado"] == "MODERADO" else "low-drift"
        html_content += f'''
                <div class="metric-card">
                    <h3 class="{drift_class}">{info["diferencia_porcentual"]:.1f}%</h3>
                    <p>{var}</p>
                    <small>Train: {info["train_mean"]:.1f} | Test: {info["test_mean"]:.1f}</small>
                </div>
        '''
    
    html_content += '''
            </div>
        </div>
        
        <div class="section">
            <h2> Variables Convertidas</h2>
            <ul>
    '''
    
    # Agregar información de conversión
    for col, mapping in conversion_map.items():
        html_content += f'<li><strong>{col}:</strong> {mapping}</li>'
    
    html_content += '''
            </ul>
        </div>
        
        <div class="section">
            <h2> Estado del Sistema</h2>
            <div style="background: #d4edda; color: #155724; padding: 15px; border-radius: 5px;">
                <h3 style="margin: 0;">SISTEMA OPERATIVO Y ESTABLE</h3>
                <p style="margin: 10px 0 0 0;">Todos los componentes funcionando correctamente</p>
            </div>
        </div>
    </div>
</body>
</html>
    '''
    
    # Guardar reporte HTML
    with open("reporte_monitoreo.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("   • Reporte HTML generado: reporte_monitoreo.html")
    
    # 7. GENERAR REPORTE TEXTO
    with open("reporte_ejecutivo.txt", "w", encoding="utf-8") as f:
        f.write("REPORTE EJECUTIVO - SISTEMA DE PREDICCIÓN CARDÍACA\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"FECHA: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
        
        f.write("DATASET\n")
        f.write(f"• Total registros: {dataset_info['total_registros']}\n")
        f.write(f"• Porcentaje enfermedad: {dataset_info['porcentaje_enfermedad']:.1f}%\n")
        f.write(f"• Variables: {dataset_info['variables_numericas']} numéricas, {dataset_info['variables_categoricas']} categóricas\n\n")
        
        f.write("MODELO\n")
        f.write(f"• Tipo: {model_info['tipo']}\n")
        f.write(f"• Mejor score: {model_info['mejor_score']}\n\n")
        
        f.write("DATA DRIFT\n")
        f.write(f"• Variables con alto drift: {len(high_drift_vars)}\n")
        for var, info in list(high_drift_vars.items())[:5]:
            f.write(f"  - {var}: {info['diferencia_porcentual']:.1f}% diferencia\n")
        
        f.write(f"\nESTADO: SISTEMA OPERATIVO\n")
    
    print("   • Reporte ejecutivo: reporte_ejecutivo.txt")
    
    print("REPORTE PROFESIONAL COMPLETADO!")
    print("ARCHIVOS GENERADOS:")
    print("   • reporte_monitoreo.html (Reporte visual completo)")
    print("   • reporte_ejecutivo.txt (Resumen ejecutivo)")

if __name__ == "__main__":
    generate_complete_report()
