from src import datacleaning as dc
from src import featureEngineering as fe
from src import preprocessing as p
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from pathlib import Path
def main():
        path = r"data/raw/paisabazaar.csv"
        # Data Cleaning methods
        df=dc.load_data(path)
        df=dc.clean_data(df)

        # Feature Engineering Method
        df=fe.create_features(df)

        # Seperating Features and Target
        X=df.drop(
            'Credit_Score',
            axis=1
        )

        y=df['Credit_Score']

        # Encoding Target
        le=LabelEncoder()
        y=le.fit_transform(y)



        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        preprocessor_obj, num_cols, cat_cols = p.create_preprocesser(X_train)
        X_train_processed = preprocessor_obj.fit_transform(X_train)
        X_test_processed = preprocessor_obj.transform(X_test)


        rf=rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        rf.fit(
            X_train_processed,
            y_train
        )
        print("✓ Model training completed")

        pred=rf.predict(
            X_test_processed
        )


        # Evaluation Metrics
        accuracy=accuracy_score(
            y_test,
            pred
        )

        precision = precision_score(
            y_test,
            pred,
            average='weighted'
        )

        recall=recall_score(
            y_test,
            pred,
            average='weighted'
        )

        f1=f1_score(
            y_test,
            pred,
            average='weighted'
        )
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")


        print(
            classification_report(
                y_test,
                pred
            )
        )

        cm=confusion_matrix(
            y_test,
            pred
        )
        print("Confusion Matrix")
        print(cm)

        MODEL_DIR=Path("models")
        MODEL_DIR.mkdir(
            exist_ok =True
        )
        joblib.dump(
            rf,
            MODEL_DIR/"random_forest.pkl"
        )
        print("✓ random_forest.pkl saved")

        joblib.dump(
            preprocessor_obj,
            MODEL_DIR / "preprocessor.pkl"
        )
        print("✓ preprocessor.pkl saved")

        joblib.dump(
            le,
            MODEL_DIR/"label_encoder.pkl"
        )
        print("✓ label_encoder.pkl saved")
        print("\nAll artifacts saved successfully!")

if __name__=="__main__":
    main()

