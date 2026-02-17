
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from business.roi import perform_sensitivity_analysis

def test_sensitivity_analysis():
    # Create dummy data
    df = pd.DataFrame({
        'customer_id': range(100),
        'predicted_clv': np.random.uniform(100, 1000, 100),
        'segment_label': np.random.choice(['Low', 'Medium', 'High'], 100)
    })
    
    base_params = {
        'budget': 5000,
        'cost_per_action': 10,
        'conversion_rate': 0.1,
        'uplift_factor': 1.1
    }
    
    print("Running sensitivity analysis...")
    results = perform_sensitivity_analysis(df, base_params)
    
    # Check if we have results for all keys
    expected_keys = ['conversion_rate', 'uplift_factor', 'cost_per_action', 'budget']
    for key in expected_keys:
        if key not in results:
            print(f"FAILED: Missing key {key}")
            return
        
        # Check if DataFrame is not empty
        if results[key].empty:
            print(f"FAILED: Empty result for {key}")
            return
            
        # Check columns
        if 'Value' not in results[key].columns or 'ROI' not in results[key].columns:
             print(f"FAILED: Missing columns in {key}")
             return
             
    print("SUCCESS: Sensitivity analysis structure is correct.")
    print("Conversion Rate Head:")
    print(results['conversion_rate'].head())

if __name__ == "__main__":
    test_sensitivity_analysis()
