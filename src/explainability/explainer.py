import shap
import numpy as np
import pandas as pd
import joblib
import json


def load_explainer():
    """Load saved SHAP explainer, model and feature names."""
    explainer = joblib.load('src/models/shap_explainer.pkl')
    scaler = joblib.load('src/models/scaler.pkl')

    with open('src/models/feature_names.json', 'r') as f:
        feature_names = json.load(f)

    return explainer, scaler, feature_names


def get_shap_values(explainer, X_scaled):
    """Calculate SHAP values for given scaled input."""
    shap_values = explainer.shap_values(X_scaled)

    # handle both 2D and 3D shap output formats
    if len(np.array(shap_values).shape) == 3:
        return shap_values[:, :, 1]
    return shap_values[1]


def get_top_drivers(shap_values_row, feature_names, top_n=5):
    """Return top N churn drivers for a single customer."""
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values_row
    }).sort_values('SHAP Value', ascending=False)

    return shap_df.head(top_n).reset_index(drop=True)