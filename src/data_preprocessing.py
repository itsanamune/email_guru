import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """
    Load the CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def clean_data(df):
    """
    Clean the data by handling missing values and normalizing formats.
    """
    # Handle missing values
    df = df.dropna()
    
    # Convert timestamps to datetime objects
    df['send_time'] = pd.to_datetime(df['send_time'])
    
    # Extract hour of day and day of week
    df['hour_of_day'] = df['send_time'].dt.hour
    df['day_of_week'] = df['send_time'].dt.dayofweek
    
    # Calculate CTR
    df['ctr'] = df['clicks'] / df['opens']
    
    return df

def extract_features(df):
    """
    Extract key features from the dataset.
    """
    features = df[['hour_of_day', 'day_of_week', 'subject', 'opens', 'clicks', 'ctr']]
    return features

def main():
    file_path = 'data/email_campaign_data.csv'
    df = load_data(file_path)
    
    if df is not None:
        df_cleaned = clean_data(df)
        features = extract_features(df_cleaned)
        
        print("Data preprocessing completed.")
        print(f"Extracted features shape: {features.shape}")
        
        # Save preprocessed data
        features.to_csv('data/preprocessed_data.csv', index=False)
        print("Preprocessed data saved to 'data/preprocessed_data.csv'")

if __name__ == "__main__":
    main()