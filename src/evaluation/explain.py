import shap
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def calculate_shap(model, X_train, X_test):
    """
    Calculates SHAP values for the model.
    """
    print("Calculating SHAP values...")
    
    # TreeExplainer is best for XGBoost
    # Fix for "could not convert string to float" error in newer XGBoost/SHAP versions
    # We pass the model object directly. If that fails, we can try saving/loading or passing the booster.
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"Standard TreeExplainer init failed: {e}. Retrying with model.get_booster()...")
        explainer = shap.TreeExplainer(model.get_booster())
    
    # Calculate SHAP values for the test set
    # check_additivity=False to avoid issues with some xgb versions/configs
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    
    return explainer, shap_values

def save_shap_artifacts(explainer, shap_values, X_test, feature_names):
    """
    Saves SHAP artifacts for the dashboard.
    """
    os.makedirs('reports', exist_ok=True)
    
    # 1. Save SHAP values (numpy array)
    with open('models/shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values, f)
        
    print("Saved SHAP values to models/shap_values.pkl")
    
    # 2. Global Importance (Mean |SHAP|)
    # Use numpy directly on the matrix
    if isinstance(shap_values, list): # For multi-class, but here we regress
        vals = shap_values[0]
    else:
        vals = shap_values
        
    global_importance = np.abs(vals).mean(0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': global_importance
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv('reports/shap_importance.csv', index=False)
    print("Saved SHAP global importance to reports/shap_importance.csv")
    
