import pandas as pd
import numpy as np



def load_data(path):
    
    return pd.read_csv(path)


def missing_values(df):
    return df.isnull.sum()


def remove_duplicates(df):

    return df.drop_duplicates()

def clean_data(df):
    df=remove_duplicates(df)
    return df

