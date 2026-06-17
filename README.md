# Paisabazaar Credit Score Prediction 🏦📊

A machine learning project that predicts **credit score categories** using customer financial data.  
It includes a **Streamlit dashboard** for interactive predictions and is designed with clean modular code for deployment.

---

## 🚀 Features
- **Data Preprocessing**: Automated cleaning and feature engineering pipeline.
- **Model Training**: Random Forest classifier trained on the Paisabazaar dataset.
- **Prediction API**: Modular `predict.py` script that loads models and preprocessing objects.
- **Interactive Dashboard**: Streamlit app (`app/app.py`) for real-time predictions.
- **External Model Hosting**: Large `.pkl` files stored on Google Drive, fetched at runtime using `gdown`.

---

## 📂 Project Structure
Paisabazaar-Credit-Score-Prediction/
│
├── app/
│   └── app.py                # Streamlit entry point
├── src/
│   ├── featureEngineering.py # Feature creation logic
│   ├── predict.py            # Prediction pipeline
│   ├── train_model.py        # Model training script
│   └── datacleaning.py       # Data cleaning utilities
├── data/
│   ├── raw/paisabazaar.csv   # Raw dataset
│   └── processed/            # Processed data
├── notebooks/
│   └── Paisabazaar Classification.ipynb # Experimentation notebook
├── requirements.txt          # Dependencies
├── .gitignore                # Ignore large files (.pkl, cache, etc.)
└── README.md                 # Project documentation


---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/vikaskesa/Paisabazaar-Credit-Score-Prediction.git
cd Paisabazaar-Credit-Score-Prediction
pip install -r requirements.txt
streamlit run app/app.py
