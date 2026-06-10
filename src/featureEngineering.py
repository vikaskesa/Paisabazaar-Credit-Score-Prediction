import numpy as np
import pandas as pd




def create_features(df):
    df=df.copy()
    
    df['Debt_income_ratio']=(
        df['Outstanding_Debt'] /df['Annual_Income'].replace(0,np.nan)
    ).fillna(0)

    df['Payment_Risk']= (
        df['Delay_from_due_date'] * df['Num_of_Delayed_Payment']
    ).fillna(0)

    df['EMI_Burden_Ratio']=(
        df['Total_EMI_per_month'] / df['Monthly_Inhand_Salary'].replace(0,np.nan)
    ).fillna(0)

    df['Inquiry_Risk']= (df['Num_Credit_Inquiries'] * df['Num_of_Loan'])

    df['Credit_Exposure'] = (df['Num_Credit_Card'] +df['Num_of_Loan'])

    return df