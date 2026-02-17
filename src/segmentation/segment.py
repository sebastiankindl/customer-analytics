import pandas as pd
import numpy as np

def assign_segments(df, pred_col='predicted_clv', recency_col='recency'):
    """
    Assigns business segments to customers.
    """
    # 1. Value Segmentation (Median Split on Predicted CLV)
    median_val = df[pred_col].median()
    df['value_segment'] = np.where(df[pred_col] >= median_val, 'High Value', 'Low Value')
    
    # 2. Risk Segmentation (Median Split on Recency)
    # Higher recency = More risk
    median_rec = df[recency_col].median()
    df['risk_segment'] = np.where(df[recency_col] >= median_rec, 'At Risk', 'Healthy')
    
    # 3. Combined Segment
    df['segment_label'] = df['value_segment'] + ' - ' + df['risk_segment']
    
    # 4. Recommended Actions
    actions = {
        'High Value - Healthy': 'Upsell / Loyalty Program',
        'High Value - At Risk': 'Retention Priority / Win-back',
        'Low Value - Healthy': 'Nurture / Cross-sell',
        'Low Value - At Risk': 'Low Priority'
    }
    
    df['recommended_action'] = df['segment_label'].map(actions)
    
    return df
