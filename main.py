from functions import data_cleaner, codif_y_ligar
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
        
    df = data_cleaner()
    rs = input("¿Desea generar nuevamente el Profile report?(S/N)\n")
    if rs.lower()=="s":
        profile = ProfileReport(df)
        profile.to_file("output.html")
        input("\n\n\nContinuar??")

    df = df[['attendance', 'field_conditions', 'venue', 'home_team', 'away_team' ]]

    # trabajar con los datos
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]
    # Codificacion de variables
    variables_a_codificar = ["home_team", "venue", "field_conditions", "away_team"]
    for variable in variables_a_codificar:
        X = codif_y_ligar(X, variable)
    # dividir el conjunto de datos para prueba y entrenamiento
    X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, Y, test_size = 0.1, random_state = 1)

    #Elimina datos atipicos
    outlier_detector = EllipticEnvelope(contamination=0.1)
    outlier_detector.fit(X_entreno.assign(target=y_entreno))
    outlier_mask = outlier_detector.predict(X_prueba.assign(target=y_prueba)) == 1

    X_test_inliers = X_prueba[outlier_mask]
    y_test_inliers = y_prueba[outlier_mask]

    # Fit a linear regression model to the inliers in the training set
    reg  = LinearRegression() 
    reg.fit(X_entreno[outlier_detector.predict(X_entreno.assign(target=y_entreno)) == 1],
                                y_entreno[outlier_detector.predict(X_entreno.assign(target=y_entreno)) == 1])

    rcuadrado = reg.score(X_test_inliers, y_test_inliers)
    #entrenar el modelo de regresion lineal multiple
    poly_regresor = LinearRegression()
    # obtener predicciones de la regresion lineal multiple
    # poly_regresor.fit(X_entreno, y_entreno)

    #Obtener R cuadrado con los datos de prueba
    # r_squared = poly_regresor.score(X_prueba, y_prueba)

    predicciones = reg.predict(X_entreno)

    # Crea la gráfica de la regresión lineal múltiple
    plt.scatter(predicciones, y_entreno)
    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.title('Gráfica de la regresión lineal múltiple')
    #plt.show()
    #regresor = LinearRegression()
    #regresor.fit(X_entreno, y_entreno)
    print(f"\n\n\nR^2: {rcuadrado}")
    choosen_team = pd.read_excel("./data/choosen_teams.xlsx")
    pd.concat([X, Y], axis = 1).to_excel("./data/data_excel.xlsx", index=False)

    y_pred = reg.predict(choosen_team)
    print(f"Predicción [Houston Astros v. Cleveleand Indians]: {y_pred[0]}")
    print(f"Coeficientes: {reg.coef_}")

if __name__== "__main__":
    main()
