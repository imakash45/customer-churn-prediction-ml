import pandas as pd
import numpy as np


def preprocess_input(customer_dict):
    """
    Preprocess raw customer input from Streamlit app
    into model-ready format.
    
    Parameters:
        customer_dict: dictionary of raw customer inputs
    
    Returns:
        processed dataframe ready for scaling and prediction
    """
    df = pd.DataFrame([customer_dict])

    # fix Total Charges if missing or zero tenure
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

    # simplify service columns
    service_cols = ['Multiple Lines', 'Online Security', 'Online Backup',
                    'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    # tenure group
    def tenure_group(tenure):
        if tenure <= 12:
            return 'New'
        elif tenure <= 24:
            return 'Developing'
        elif tenure <= 48:
            return 'Established'
        else:
            return 'Loyal'

    df['Tenure Group'] = df['Tenure Months'].apply(tenure_group)

    # spending features
    df['Avg Monthly Spend'] = df['Total Charges'] / (df['Tenure Months'] + 1)
    df['Charge to CLTV Ratio'] = df['Monthly Charges'] / df['CLTV']

    # total services
    service_cols_count = ['Phone Service', 'Multiple Lines', 'Online Security',
                          'Online Backup', 'Device Protection', 'Tech Support',
                          'Streaming TV', 'Streaming Movies']
    df['Total Services'] = df[service_cols_count].apply(lambda x: (x == 'Yes').sum(), axis=1)

    # risk indicators
    df['Is Month to Month'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['Is Electronic Check'] = (df['Payment Method'] == 'Electronic check').astype(int)
    df['Is Fiber Optic'] = (df['Internet Service'] == 'Fiber optic').astype(int)
    df['Is Senior'] = (df['Senior Citizen'] == 'Yes').astype(int)
    df['No Tech Support'] = (df['Tech Support'] == 'No').astype(int)
    df['No Online Security'] = (df['Online Security'] == 'No').astype(int)

    # value segment
    def value_segment(cltv):
        if cltv <= 3469:
            return 'Low Value'
        elif cltv <= 5380:
            return 'Medium Value'
        else:
            return 'High Value'

    df['Value Segment'] = df['CLTV'].apply(value_segment)

    # binary encoding
    binary_cols = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
                   'Multiple Lines', 'Online Security', 'Online Backup', 'Device Protection',
                   'Tech Support', 'Streaming TV', 'Streaming Movies', 'Paperless Billing']
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col].str.strip() == 'Yes').astype(int)

    # one hot encoding
    df = pd.get_dummies(df, columns=['Internet Service', 'Contract',
                                      'Payment Method', 'Tenure Group',
                                      'Value Segment'], drop_first=False)

    # convert booleans to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def align_features(df, feature_names):
    """
    Align input dataframe columns with training feature names.
    Adds missing columns as 0 and removes extra columns.
    
    Parameters:
        df: preprocessed dataframe
        feature_names: list of feature names from training
    
    Returns:
        dataframe with exact same columns as training data
    """
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]