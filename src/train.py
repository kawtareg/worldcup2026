from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def prepare_data(df):
    train = df[df['date'] < '2020-01-01']
    test = df[df['date'] >= '2020-01-01']
    
    X_train = train[['home_form', 'away_form', 'elo_diff']]
    X_test = test[['home_form', 'away_form', 'elo_diff']]
    
    y_train = train['result'].map({'W': 0, 'D': 1, 'L': 2})
    y_test = test['result'].map({'W': 0, 'D': 1, 'L': 2})
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    print('Log loss: ', log_loss(y_test, y_pred))
    return model
