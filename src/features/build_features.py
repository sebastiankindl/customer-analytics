import pandas as pd
import numpy as np
from datetime import timedelta
import os

def load_data():
    """Lengths raw data."""
    if not os.path.exists('data/transactions.csv'):
        raise FileNotFoundError("Run src/data/make_dataset.py first")
    
    df = pd.read_csv('data/transactions.csv')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

def create_features(df, observation_date, prediction_window_days=90):
    """
    Creates features for customers based on history up to observation_date.
    Target is revenue in [observation_date, observation_date + prediction_window_days].
    """
    
    # 1. Split history and future
    history = df[df['transaction_date'] <= observation_date].copy()
    future = df[(df['transaction_date'] > observation_date) & 
                (df['transaction_date'] <= observation_date + timedelta(days=prediction_window_days))].copy()
    
    # If no history, we can't make predictions for existing customers (or they are new)
    # We focus on existing customers in history
    cust_history = history.groupby('customer_id')
    
    # --- Feature Engineering ---
    
    # RFM
    recency = (observation_date - cust_history['transaction_date'].max()).dt.days
    frequency = cust_history['transaction_date'].count()
    monetary = cust_history['amount'].sum()
    
    # Behavioral
    tenure = (observation_date - cust_history['transaction_date'].min()).dt.days
    aov = monetary / frequency
    
    features = pd.DataFrame({
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'tenure': tenure,
        'aov': aov
    })
    
    # Complex features (e.g. inter-purchase time)
    # Average time between purchases
    def avg_inter_purchase_time(x):
        if len(x) < 2:
            return 0
        return (x.max() - x.min()).days / (len(x) - 1)
        
    features['avg_inter_purchase_time'] = cust_history['transaction_date'].apply(avg_inter_purchase_time)
    
    # --- Target Creation ---
    
    future_revenue = future.groupby('customer_id')['amount'].sum()
    features['target_revenue'] = future_revenue
    features['target_revenue'] = features['target_revenue'].fillna(0)
    
    return features

def run_feature_pipeline():
    print("Loading data...")
    df = load_data()
    
    # Define Split Dates
    # We want to simulate a production scenario.
    # Train: Observe up to T_train, predict T_train + 90d
    # Test: Observe up to T_test, predict T_test + 90d
    # Ensure Test window is after Train window to prevent leakage
    
    max_date = df['transaction_date'].max()
    print(f"Data range: {df['transaction_date'].min()} to {max_date}")
    
    # Test cutoff: 90 days before end of data (so we have ground truth)
    test_observation_date = max_date - timedelta(days=90)
    
    # Train cutoff: 90 days before test observation (simple non-overlapping)
    # or even earlier. Let's do 180 days before test.
    train_observation_date = test_observation_date - timedelta(days=180)
    
    print(f"Train Observation Date: {train_observation_date}")
    print(f"Test Observation Date: {test_observation_date}")
    
    print("Building Train set...")
    train_df = create_features(df, train_observation_date)
    train_df['dataset_type'] = 'train'
    
    print("Building Test set...")
    test_df = create_features(df, test_observation_date)
    test_df['dataset_type'] = 'test' # In production, this would be 'holdout' with known targets
    
    # Combine or save separately
    os.makedirs('data/processed', exist_ok=True)
    
    train_df.to_csv('data/processed/train.csv')
    test_df.to_csv('data/processed/test.csv')
    
    print(f"Saved train ({len(train_df)}) and test ({len(test_df)}) datasets.")

if __name__ == "__main__":
    run_feature_pipeline()
