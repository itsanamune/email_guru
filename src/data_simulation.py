import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_simulated_data(num_samples=1000):
    """
    Generate simulated email campaign data.
    """
    # Generate random send times
    start_date = datetime(2023, 1, 1)
    send_times = [start_date + timedelta(hours=np.random.randint(0, 24*365)) for _ in range(num_samples)]
    
    # Generate subject lines
    subjects = [f"Subject {i+1}" for i in range(num_samples)]
    
    # Generate open rates and click-through rates
    open_rates = np.random.beta(2, 5, num_samples)
    ctr = np.random.beta(1, 10, num_samples)
    
    # Calculate opens and clicks
    recipients = np.random.randint(1000, 10000, num_samples)
    opens = (recipients * open_rates).astype(int)
    clicks = (opens * ctr).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'send_time': send_times,
        'subject': subjects,
        'recipients': recipients,
        'opens': opens,
        'clicks': clicks
    })
    
    return df

def main():
    # Create the 'simulated_data' directory if it doesn't exist
    os.makedirs('simulated_data', exist_ok=True)

    simulated_data = generate_simulated_data()
    print(f"Generated simulated data. Shape: {simulated_data.shape}")
    
    # Save simulated data
    simulated_data.to_csv('simulated_data/simulated_email_data.csv', index=False)
    print("Simulated data saved to 'simulated_data/simulated_email_data.csv'")

if __name__ == "__main__":
    main()