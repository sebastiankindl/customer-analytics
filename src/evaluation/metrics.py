import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Calculates standard regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

def top_decile_lift(y_true, y_pred):
    """
    Calculates the lift in the top decile of predictions.
    How much more revenue do we capture by targeting the top 10% predicted vs random?
    """
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df = df.sort_values('pred', ascending=False)
    
    top_10_pct_n = int(len(df) * 0.1)
    if top_10_pct_n == 0:
        return 0.0
        
    captured_revenue = df.iloc[:top_10_pct_n]['true'].sum()
    total_revenue = df['true'].sum()
    random_revenue = total_revenue * 0.1
    
    lift = captured_revenue / random_revenue if random_revenue > 0 else 0
    capture_rate = captured_revenue / total_revenue if total_revenue > 0 else 0
    
    return lift, capture_rate
