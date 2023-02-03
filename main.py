import pandas as pd
import numpy as np
from constants import DATA_PATH
import re
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def formatt_start_time_into_float(start_time: str):
    """Transform string values of column formatted: 'Start Time: HH: MM (p.m.|a.m.) Local' and turns into float number in this interval [0, 24)

    Args:
        start_time (str): String fomatted likes this 'Start Time: HH: MM (p.m.|a.m.) Local'

    Returns:
        float: Hour turned into a float in this range [0, 24)
    """
    if not re.match(r"^\s*(Start Time:)\s?\d?\d:\d\d\s*((p.m.?)|(a.m.?))\s*Local$", start_time): return None
    
    start_time = start_time.split(" ")
    hour_min = start_time[2].split(":")
    hour = (int(hour_min[0])%12+12) if "p.m" in start_time[3] else int(hour_min[0])%12
    return hour+int(hour_min[1])/60

def formatt_game_duration_into_float(game_duration: str):
    """Transforms a string value from column game_duration formatted ': H:MM' into a number in this interval [0, inf.)

    Args:
        game_duration (str): A string value with this formatted like this ': H:MM'

    Returns:
        float: Hour turned into a float in this range [0, inf.)
    """
    if not re.match(r"^\s*:?\s*\d+:\d+\s*$", game_duration): return None

    game_duration = game_duration.split(":")
    return (int(game_duration[-1])/60)+int(game_duration[-2])


def codif_y_ligar(dataframe_original, variables_a_codificar):
    dummies = pd.get_dummies(dataframe_original[[variables_a_codificar]], prefix=variables_a_codificar)
    res = pd.concat([dataframe_original, dummies], axis = 1)
    res = res.drop([variables_a_codificar], axis = 1)
    return res 

def main():
        
    df = pd.read_csv(DATA_PATH)
    n_rows = df.shape[0]

    # 1.1 resumen del dataset
    print("resumen del dataset:")
    print(df.describe())

    df.drop(columns=["other_info_string", "boxscore_url", "field_type", "date"], inplace=True)

    # Cleaning dataset
    df['attendance'] = df['attendance'].apply(lambda x: int(x.split("'")[0].replace(",", "")) if (x.split("'")[0].replace(",", "").isnumeric()) else None)

    # Removing data with no info of the attendance
    df = df.dropna(subset=["attendance"])
    n_rows = n_rows - df.shape[0]
    print(f'Rows removed because of null values: {n_rows}')

    df['start_time'] = df['start_time'].apply(formatt_start_time_into_float)
    df['game_duration'] = df['game_duration'].apply(formatt_game_duration_into_float)

    pattern = re.compile(r'^[^\d]+$')
    df["venue"] = df["venue"].apply(lambda s: None if not pattern.match(s) else s.replace(":", "").replace(" ", "").lower()) 

    df[["time_condition", "field_conditions"]] = df['game_type'].str.split(',', expand=True)
    df.drop(columns=["game_type"], inplace=True)

    #1.2
    print("tipos de variables (columnas):")
    for column in df:
        print(column, df[column].dtype)
    
    print("\n Dataset:")
    print(df)

    # trabajar con los datos
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    # codificacion de variables
    variables_a_codificar = ["away_team", "home_team", "venue", "time_condition", "field_conditions"]
    for variable in variables_a_codificar:
        X = codif_y_ligar(X, variable)

    # dividir el conjunto de datos para prueba y entrenamiento
    X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, Y, test_size = 0.2, random_state = 1)

    #entrenar el modelo de regresion lineal multiple
    poly_regresor = LinearRegression()
    # obtener predicciones de la regresion lineal multiple
    poly_regresor.fit(X_entreno, y_entreno)

    #Obtener R cuadrado con los datos de prueba
    r_squared = poly_regresor.score(X_prueba, y_prueba)
    print("R^2:", r_squared)

    predicciones = poly_regresor.predict(X_entreno)

    # Crea la gráfica de la regresión lineal múltiple
    plt.scatter(predicciones, y_entreno)
    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.title('Gráfica de la regresión lineal múltiple')
    plt.show()

    #Parece que la regresion lineal no es la adecuada :p
    #polinomial
    regresor_poli = PolynomialFeatures(degree = 2)
    X_poli = regresor_poli.fit_transform(X_entreno)

    # obtener predicciones de la regresion lineal multiple
    regresor = LinearRegression()
    regresor.fit(X_poli, y_entreno)
    predicciones = regresor.predict(X_poli)

    #Obtener R cuadrado con los datos de prueba
    r_squared = regresor.score(X_prueba, y_prueba)
    print("R^2:", r_squared)
     # Crea la gráfica de la regresión lineal múltiple
    plt.scatter(predicciones, y_entreno)
    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.title('Gráfica de la regresión lineal múltiple')
    plt.show()

    
    
    #df.to_excel("./data/data_excel.xlsx", index=False)

if __name__== "__main__":
    main()
