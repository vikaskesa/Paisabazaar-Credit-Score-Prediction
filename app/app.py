import streamlit as st
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_credit_score


import sys
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
def format_loans(loans):
    if len(loans)==0:
        return "No Data"
    
    if len(loans)==1:
        return loans[0]
    elif(len(loans)==2):
        return f"{loans[0]} and {loans[1]}"
    else:
        return (
            ", ".join(loans[:-1])
            +", and "
            +loans[-1]
        )
# Unique Elements in Categorical columns    
Occupation_var = ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect']
Type_of_Loan_var =['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan', 'No Data', 'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']
Credit_Mix_var = ['Good', 'Standard', 'Bad']
Payment_of_Min_Amount_var =['No', 'NM', 'Yes']
Payment_Behaviour_var =['High_spent_Small_value_payments', 'Low_spent_Large_value_payments', 'Low_spent_Medium_value_payments', 'Low_spent_Small_value_payments', 'High_spent_Medium_value_payments', 'High_spent_Large_value_payments']

st.set_page_config(
    page_title='Credit Score Predictor',
    layout='wide'
)
st.title("💳 Paisabazaar Credit Score Classification")


st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #198754;
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    padding: 12px 24px;
    width: 100%;
}

div.stButton > button:first-child:hover {
    background-color: #157347;
    color: white;
    transform: scale(1.02);
    transition: 0.2s;
}
</style>
""", unsafe_allow_html=True)



col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Personal Information")
    Age=st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    Month =st.number_input(
        "Month",
        min_value=1,
        max_value=12
    )

    Occupation=st.selectbox(
        "Occupation",
        Occupation_var
    )

    st.subheader("💰 Income & Financial Information")

    Annual_Income=st.number_input(
        "Annual Income",
        min_value=0.0
    )

    Monthly_Inhand_Salary=st.number_input(
        "Monthly Inhand Salary",
        min_value=0.0

    )

    Amount_invested_monthly=st.number_input(
        "Monthly Invested Amount",
        min_value=0.0
    )

    Monthly_Balance=st.number_input(
        "Monthly Balance",
        min_value=0.0
    )
    st.subheader("⚠️ Credit Behaviour & Payment History")
    
    Delay_from_due_date=st.number_input(
    "Delay from Due Date (in Days)",
    min_value=0,
    max_value=365
    )

    Num_of_Delayed_Payment=st.number_input(
    "Number of Delayed Payments",
    min_value=0,
    max_value=100
    )

    Num_Credit_Inquiries=st.number_input(
    "Number of Credit Inquiries",
    min_value=0,
    max_value=25
    )
    Payment_of_Min_Amount=st.selectbox(
    "Payment of Minimum Amount",
    Payment_of_Min_Amount_var
    )   

    Payment_Behaviour=st.selectbox(
        "Payment Behaviour",
        Payment_Behaviour_var
    )


with col2:
    st.subheader("🏦 Banking & Credit Profile")
    Num_Bank_Accounts=st.number_input(
    "Number of Bank Accounts",
    min_value=1,
    max_value=20,
    )
    Num_Credit_Card=st.number_input(
    "Number of Credit Cards",
    min_value=1,
    max_value=20
    )
    Credit_Mix = st.selectbox(
    "Credit Mix",
    Credit_Mix_var
    )
    Credit_History_Age=st.number_input(
    "Credit History Age",
    min_value=0
    )
    Credit_Utilization_Ratio=st.number_input(
    "Credit Utilization Ratio (in Percent)",
    min_value=0.0,
    max_value=100.0
    )
    Changed_Credit_Limit=st.number_input(
    "Changed Credit Limit ( In Percent )",
    min_value=0.00,
    max_value=100.00
    )
    st.subheader("📋 Loan Information")
    
    Num_of_Loan =int(st.number_input(
    "Number of loans ",
    min_value=0,
    max_value=20
    ))

    loan_inputs=[]
    for i in range(Num_of_Loan):
        loan=st.selectbox(
            f"Loan {i+1}",
            Type_of_Loan_var,
            key=f"Loan_{i}"
        )
        loan_inputs.append(loan)

    loans=format_loans(loan_inputs)
    Interest_Rate=st.number_input(
    "Interest Rate(in Percent)",
    min_value =0.0,
    max_value=100.0
    )
    
    Outstanding_Debt=st.number_input(
    "Outstanding Debt",
    min_value=0.0
    )

    Total_EMI_per_month=st.number_input(
    "Total EMI Per Month",
    min_value=0.0
    )

sample={
        "Month": Month,
    "Age": Age,
    "Occupation": Occupation,                          
    "Annual_Income": Annual_Income,
    "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
    "Num_Bank_Accounts": Num_Bank_Accounts,
    "Num_Credit_Card": Num_Credit_Card,
    "Interest_Rate": Interest_Rate,
    "Num_of_Loan": Num_of_Loan,
    "Type_of_Loan": loans,                         
    "Delay_from_due_date": Delay_from_due_date,
    "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
    "Changed_Credit_Limit": Changed_Credit_Limit,
    "Num_Credit_Inquiries": Num_Credit_Inquiries,
    "Credit_Mix":Credit_Mix,
    "Outstanding_Debt": Outstanding_Debt,
    "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
    "Credit_History_Age": Credit_History_Age,
    "Payment_of_Min_Amount": Payment_of_Min_Amount,
    "Total_EMI_per_month": Total_EMI_per_month,
    "Amount_invested_monthly": Amount_invested_monthly,
    "Payment_Behaviour": Payment_Behaviour,
    "Monthly_Balance": Monthly_Balance
    }
st.markdown("---")
st.subheader("🎯 Credit Score Prediction")
if(st.button("Predict Credit Score")):
    inp_df=pd.DataFrame([sample])
    with st.spinner(
        "Analyzing Customer Profile.."
    ):
        result=predict_credit_score(inp_df)

    if result=="Good":
        st.success(
            f"✅ Predicted Credit Score: {result}"
        )
    elif result=="Standard":
        st.warning(
             f"⚠️ Predicted Credit Score: {result}"
        )
    else:
        st.error(
            f"❌ Predicted Credit Score: {result}"
        )



