import pandas as pd
from constants import DATA_PATH
import re
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    variables_a_codificar = ["away_team", "home_team", "venue", "time_condition", "field_conditions"]
    for variable in variables_a_codificar:
        X = codif_y_ligar(X, variable)
    
    X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, Y, test_size = 0.2, random_state = 1)
    
    

   
    
    df.to_excel("./data/data_excel.xlsx", index=False)

if __name__== "__main__":
    main()
