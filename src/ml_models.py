import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def load_and_prepare_data():
    """
    Load the preprocessed data and prepare it for model training.
    """
    df = pd.read_csv('data/preprocessed_data.csv')
    return df

def train_send_time_model(df):
    """
    Train a model to predict optimal send times.
    """
    X = df[['hour_of_day', 'day_of_week']]
    y = df[['opens', 'clicks']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Send Time Model MSE: {mse}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/send_time_model.joblib')
    print("Send Time Model saved to 'models/send_time_model.joblib'")

    return model

def train_subject_line_model(df):
    """
    Train a model to recommend subject lines.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['subject'])
    y = df[['opens', 'clicks']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Subject Line Model MSE: {mse}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/subject_line_model.joblib')
    joblib.dump(vectorizer, 'models/subject_line_vectorizer.joblib')
    print("Subject Line Model and Vectorizer saved to 'models/' directory")

    return model, vectorizer

def main():
    df = load_and_prepare_data()
    send_time_model = train_send_time_model(df)
    subject_line_model, vectorizer = train_subject_line_model(df)

if __name__ == "__main__":
    main()