import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(num_samples=10000):
    # Generate random send times
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    send_times = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(num_samples)]

    # Generate subject lines
    subjects = [
        f"Limited time offer: {np.random.randint(10, 75)}% off!",
        "New arrivals just for you",
        f"Don't miss out on these {np.random.choice(['amazing', 'incredible', 'fantastic'])} deals",
        f"Your {np.random.choice(['weekly', 'monthly', 'exclusive'])} newsletter",
        f"{np.random.choice(['Summer', 'Winter', 'Spring', 'Fall'])} sale starts now!",
    ]

    # Generate data
    data = {
        'send_time': send_times,
        'subject': [np.random.choice(subjects) for _ in range(num_samples)],
        'recipients': np.random.randint(1000, 100000, num_samples),
        'opens': np.random.randint(100, 50000, num_samples),
        'clicks': np.random.randint(10, 5000, num_samples),
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure opens and clicks are not greater than recipients
    df['opens'] = df.apply(lambda row: min(row['opens'], row['recipients']), axis=1)
    df['clicks'] = df.apply(lambda row: min(row['clicks'], row['opens']), axis=1)

    return df

def main():
    # Create the 'data' directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    df = generate_sample_data()
    df.to_csv('data/email_campaign_data.csv', index=False)
    print(f"Sample data generated and saved to 'data/email_campaign_data.csv'")
    print(f"Shape of the generated data: {df.shape}")
    print("\nFirst few rows of the generated data:")
    print(df.head())

if __name__ == "__main__":
    main()