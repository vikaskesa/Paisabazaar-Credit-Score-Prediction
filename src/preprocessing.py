from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

def create_preprocesser(df):
    num_cols =df.select_dtypes(
        include=['int64','float64']
    ).columns.tolist()
    if 'Credit_Score' in num_cols:
        num_cols.remove('Credit_Score')

    cat_cols=df.select_dtypes(
        include='object'
    ).columns.tolist()

    preprocessor=ColumnTransformer(
        transformers=[
            (
                'num',
                StandardScaler(),
                num_cols
            ),
            (
                'cat',
                OneHotEncoder(
                    handle_unknown='ignore'
                ),
                cat_cols
            )
        ]
    )
    return preprocessor,num_cols,cat_cols