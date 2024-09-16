import joblib
import pandas as pd

def load_models():
    """
    Load the trained models from the saved file.
    """
    models = joblib.load('well_log_v1.pkl')  
    return models

def predict_well_logs(well_data):
    """
    Predict well logs (NPHI, log_RD, RHOB) sequentially using the trained models.
    
    Args:
    well_data (pd.DataFrame): Input data with columns ['Pseudo_TVD', 'GR'].
    
    Returns:
    pd.DataFrame: Input data with additional columns for predictions.
    """
    models = load_models()

    # Ensure the data has the required columns
    required_columns = ['Pseudo_TVD', 'GR']
    for col in required_columns:
        if col not in well_data.columns:
            raise ValueError(f"Missing required column: {col}")

    # 1st model - Predict NPHI
    x_feat1 = well_data[['Pseudo_TVD', 'GR']]
    if x_feat1.isnull().values.any():
        raise ValueError("Input data contains missing values")
    well_data['NPHI_pred'] = models[0][0].predict(x_feat1)

    # 2nd model - Predict log_RD
    well_data['NPHI_pred'] = well_data['NPHI_pred'].fillna(method='ffill')  # Handle any potential NaNs
    x_feat2 = well_data[['Pseudo_TVD', 'GR', 'NPHI_pred']]
    if x_feat2.isnull().values.any():
        raise ValueError("Input data contains missing values after first prediction")
    well_data['log_RD_pred'] = models[0][1].predict(x_feat2)

    # 3rd model - Predict RHOB
    well_data['log_RD_pred'] = well_data['log_RD_pred'].fillna(method='ffill')  # Handle any potential NaNs
    x_feat3 = well_data[['Pseudo_TVD', 'GR', 'NPHI_pred', 'log_RD_pred']]
    if x_feat3.isnull().values.any():
        raise ValueError("Input data contains missing values after second prediction")
    well_data['RHOB_pred'] = models[0][2].predict(x_feat3)

    return well_data
