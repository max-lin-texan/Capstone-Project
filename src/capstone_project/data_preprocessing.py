from file_read import load_data

def data_preprocessing():
    df_sepsis_vitals = load_data()

    return df_sepsis_vitals

df = data_preprocessing()
print(df.head())