import pandas as pd
from constants import DATA_PATH
import re

def main():
    df = pd.read_csv(DATA_PATH)
    
    
    def get_start_time(start_time):
        start_time = start_time.split(" ")
        hour_min = start_time[2].split(":")

        hour = (int(hour_min[0])%12+12) if "p.m" in start_time[3] else int(hour_min[0])%12

        return f"{hour}:{hour_min[1]}"

    
    df['attendance'] = df['attendance'].apply(lambda x: int(x.split("'")[0].replace(",", "")) if (x.split("'")[0].replace(",", "").isnumeric()) else None)
    df['start_time'] = df['start_time'].apply(get_start_time)
    
    
    # print(df[pd.isnull(df["attendance"])])
    # df.to_excel("./data/data_excel.xlsx", index=False)

if __name__== "__main__":
    main()
