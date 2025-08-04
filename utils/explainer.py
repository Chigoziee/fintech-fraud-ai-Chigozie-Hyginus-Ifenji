import shap
import pandas as pd
from utils.predict import model

# Load the SHAP explainer once to reuse
explainer = shap.Explainer(model)

def explain_prediction(data: pd.DataFrame) -> dict:
    shap_values = explainer(data)
    feature_importance = dict(zip(data.columns, shap_values.values[0]))
    return feature_importance, shap_values
