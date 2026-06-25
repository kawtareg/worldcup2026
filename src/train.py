"""Functions to train and evaluate the XGBoost match outcome prediction model."""

import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

def prepare_data(df):
    """Split dataframe into train and test sets with features and target variable.

    Args:
        df (pd.DataFrame): Match dataframe with features and result column.

    Returns:
        tuple: X_train, X_test, y_train, y_test as pandas DataFrames/Series.
    """
    train = df[df['date'] < '2020-01-01']
    test = df[df['date'] >= '2020-01-01']

    features = ['home_form', 'away_form', 'elo_diff',
            'home_avg_scored', 'away_avg_scored',
            'home_avg_conceded', 'away_avg_conceded']

    X_train = train[features].apply(pd.to_numeric, errors='coerce')
    X_test = test[features].apply(pd.to_numeric, errors='coerce')

    y_train = train['result'].map({'W': 0, 'D': 1, 'L': 2})
    y_test = test['result'].map({'W': 0, 'D': 1, 'L': 2})

    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """Train an XGBoost classifier and save it to disk.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.

    Returns:
        XGBClassifier: Trained XGBoost model.
    """
    model = XGBClassifier(
    n_estimators=73,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    eval_metric='mlogloss'
)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict_proba(X_test)
    print('Log loss: ', log_loss(y_test, y_pred))
    path = Path(__file__).parent.parent / "models" / "xgboost.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


