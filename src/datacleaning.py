import pandas as pd
import numpy as np

path='C:\Users\kesav\Desktop\Paisabzaar Credit score Classification\data\raw\paisabazaar.csv'

def load_data(path):
    
    return pd.read_csv(path)

df=load_data(path)


def missing_values(df):
    return df.isnull.sum()


def remove_duplicates(df):

    return df.drop_duplicates()

def clean_data(df):
    df=remove_duplicates(df)
    return df

