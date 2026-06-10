import pandas as pd
import numpy as np



def load_data(path):
    return pd.read_csv(path)


def missing_values(df):
    return df.isnull().sum()


def remove_duplicates(df):
    return df.drop_duplicates()

def drop_unnecessary_cols(df):
    columns_to_drop =[
        "ID",
        'Customer_ID',
        'Name',
        'SSN'
    ]
    df=df.drop(
        columns=columns_to_drop,
        errors='ignore'
    )
    return df


def clean_data(df):
    df=remove_duplicates(df)
    df=drop_unnecessary_cols(df)
    return df

