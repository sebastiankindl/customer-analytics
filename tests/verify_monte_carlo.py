
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from business.roi import perform_monte_carlo_simulation

def test_monte_carlo():
    # Create dummy data
    df = pd.DataFrame({
        'customer_id': range(100),
        'predicted_clv': np.ones(100) * 100, # Fixed CLV for easy math
        'segment_label': ['High'] * 100
    })
    
    # Base Case: 
    # Budget=500, Cost=10 -> 50 customers.
    # Total Cost = 500.
    # Base Revenue = 50 * 100 = 5000.
    # Uplift Factor = 1.1, Conv = 0.5
    # Exp Uplift = 5000 * (1 + 0.5 * (1.1 - 1)) - 5000
    #            = 5000 * (1 + 0.5 * 0.1) - 5000
    #            = 5000 * 1.05 - 5000 = 5250 - 5000 = 250
    # Expected ROI = (250 - 500) / 500 = -0.5 (-50%)
    
    base_params = {
        'budget': 500,
        'cost_per_action': 10,
        'conversion_rate': 0.5,
        'uplift_factor': 1.1
    }
    
    std_devs = {
        'conversion_rate': 0.0, # NO uncertainty first
        'uplift_factor': 0.0
    }
    
    print("Running Deterministic MC check...")
    results = perform_monte_carlo_simulation(df, base_params, std_devs, n_iterations=10)
    
    mean_roi = results['ROI'].mean()
    print(f"Mean ROI (Deterministic): {mean_roi:.4f}")
    
    # Check if close to expected -0.5
    if abs(mean_roi - (-0.5)) > 0.001:
        print("FAILED: Deterministic ROI mismatch.")
    else:
        print("SUCCESS: Deterministic ROI matches.")
        
    # Now check stochastic
    print("\nRunning Stochastic MC check...")
    std_devs = {
        'conversion_rate': 0.1, 
        'uplift_factor': 0.1
    }
    results = perform_monte_carlo_simulation(df, base_params, std_devs, n_iterations=1000)
    
    print(f"Mean ROI: {results['ROI'].mean():.4f}")
    print(f"ROI Std Dev: {results['ROI'].std():.4f}")
    print(f"Min ROI: {results['ROI'].min():.4f}")
    print(f"Max ROI: {results['ROI'].max():.4f}")
    
    if results['ROI'].std() > 0:
        print("SUCCESS: Variation detected.")
    else:
        print("FAILED: No variation in stochastic run.")

if __name__ == "__main__":
    test_monte_carlo()
