import pandas as pd
import numpy as np


def get_risk_level(churn_probability):
    """
    Convert churn probability to human readable risk level.
    
    Parameters:
        churn_probability: float between 0 and 1
    
    Returns:
        risk level string and color
    """
    if churn_probability >= 0.7:
        return 'High Risk', '#FF4B4B'
    elif churn_probability >= 0.4:
        return 'Medium Risk', '#FFA500'
    else:
        return 'Low Risk', '#00CC44'


def format_currency(value):
    """Format a number as currency string."""
    return f"${value:,.2f}"


def get_churn_summary(churn_probability, cltv):
    """
    Generate a plain english summary of customer churn risk.
    
    Parameters:
        churn_probability: float between 0 and 1
        cltv: customer lifetime value
    
    Returns:
        summary string
    """
    risk_level, _ = get_risk_level(churn_probability)
    revenue_at_risk = churn_probability * cltv

    return (
        f"This customer is at {risk_level.lower()} of churning "
        f"with a {churn_probability:.1%} probability. "
        f"Estimated revenue at risk is ${revenue_at_risk:,.2f}."
    )