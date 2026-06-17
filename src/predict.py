from pathlib import Path

import joblib
import pandas as pd
import gdown

from src import featureEngineering as fe

def load_from_drive(file_id,file_name):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url,file_name,quiet=False)
    return joblib.load(file_name)


MODEL_FILE_ID='1WqPcP93KDoujjKtUuG-A2URpAocayGAz'
PREPROCESSOR_FILE_ID='1hJQ8HQ0q2B-kx3YP2E19LKtwGEQH1VWS'
LABEL_ENCODER_FILE_ID = '14bDWABxm-hc37i28ZWPv-XicjOJbzp0Q'

model=load_from_drive(MODEL_FILE_ID,"random_forest.pkl")
preprocessor=load_from_drive(PREPROCESSOR_FILE_ID,"preprocessor.pkl")
label_encoder=load_from_drive(LABEL_ENCODER_FILE_ID,"label_encoder.pkl")


# BASE_DIR=Path(__file__).resolve().parent.parent

# MODEL_DIR=BASE_DIR / "models"

# model=joblib.load(
#     MODEL_DIR / "random_forest.pkl"
# )

# preprocessor=joblib.load(
#     MODEL_DIR/"preprocessor.pkl"
# )

# label_encoder=joblib.load(
#     MODEL_DIR / 'label_encoder.pkl'
# )

def predict_credit_score(input_df):

    input_df = fe.create_features(
        input_df
    )

    processed_data=preprocessor.transform(
        input_df
    )

    prediction = model.predict(
        processed_data
    )

    prediction_label = (
        label_encoder.inverse_transform(prediction)
    )

    return prediction_label[0]

if __name__=='__main__':
    sample={
        "Month": 6,
    "Age": 35,
    "Occupation": "Engineer",
    "Annual_Income": 1200000,
    "Monthly_Inhand_Salary": 95000,
    "Num_Bank_Accounts": 3,
    "Num_Credit_Card": 2,
    "Interest_Rate": 8,
    "Num_of_Loan": 1,
    "Type_of_Loan": "Home Loan",
    "Delay_from_due_date": 2,
    "Num_of_Delayed_Payment": 0,
    "Changed_Credit_Limit": 5.0,
    "Num_Credit_Inquiries": 1,
    "Credit_Mix": "Good",
    "Outstanding_Debt": 25000,
    "Credit_Utilization_Ratio": 22.5,
    "Credit_History_Age": 180,
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 12000,
    "Amount_invested_monthly": 15000,
    "Payment_Behaviour": "High_spent_Small_value_payments",
    "Monthly_Balance": 55000
    }
    sample_df=pd.DataFrame([sample])
    result=predict_credit_score(sample_df)
    print(result)
