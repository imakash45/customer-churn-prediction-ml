import pandas as pd


# retention rules based on EDA and SHAP findings
# each rule checks churn drivers and suggests a specific business action

def get_retention_recommendations(customer_data, top_drivers):
    """
    Generate retention recommendations based on customer's top churn drivers.
    
    Parameters:
        customer_data: dict of customer feature values
        top_drivers: dataframe of top SHAP features for this customer
    
    Returns:
        list of recommendation dictionaries
    """
    recommendations = []
    driver_features = top_drivers['Feature'].tolist()

    # rule 1 - month to month contract is top driver
    if any('Month-to-month' in f or 'Is Month to Month' in f for f in driver_features):
        recommendations.append({
            'Priority': 'High',
            'Action': 'Offer Annual Plan Upgrade',
            'Detail': 'Provide 20% discount on 1-year or 2-year contract upgrade.',
            'Expected Impact': 'Reduces churn risk by ~35%'
        })

    # rule 2 - short tenure / new customer
    if any('Tenure' in f or 'New' in f for f in driver_features):
        recommendations.append({
            'Priority': 'High',
            'Action': 'Personalized Onboarding Campaign',
            'Detail': 'Assign dedicated support agent for first 3 months.',
            'Expected Impact': 'Improves early retention by ~25%'
        })

    # rule 3 - fiber optic with high charges
    if any('Fiber' in f for f in driver_features):
        recommendations.append({
            'Priority': 'Medium',
            'Action': 'Offer Fiber Loyalty Discount',
            'Detail': 'Provide 10% bill reduction or free speed upgrade for 3 months.',
            'Expected Impact': 'Reduces price-related churn by ~20%'
        })

    # rule 4 - electronic check payment
    if any('Electronic check' in f or 'Is Electronic Check' in f for f in driver_features):
        recommendations.append({
            'Priority': 'Medium',
            'Action': 'Encourage Auto Payment Switch',
            'Detail': 'Offer $5 monthly discount for switching to auto bank transfer.',
            'Expected Impact': 'Reduces payment friction churn by ~15%'
        })

    # rule 5 - no tech support
    if any('Tech Support' in f for f in driver_features):
        recommendations.append({
            'Priority': 'Medium',
            'Action': 'Free Tech Support Trial',
            'Detail': 'Offer 2 months free tech support to demonstrate value.',
            'Expected Impact': 'Increases service stickiness by ~18%'
        })

    # rule 6 - no online security
    if any('Online Security' in f for f in driver_features):
        recommendations.append({
            'Priority': 'Low',
            'Action': 'Online Security Bundle Offer',
            'Detail': 'Add online security to plan at 50% off for 6 months.',
            'Expected Impact': 'Increases bundle attachment by ~12%'
        })

    # rule 7 - no dependents / single customer
    if any('Dependent' in f for f in driver_features):
        recommendations.append({
            'Priority': 'Low',
            'Action': 'Engagement Campaign',
            'Detail': 'Send personalized offers and loyalty rewards program invitation.',
            'Expected Impact': 'Improves engagement by ~10%'
        })

    # default recommendation if no specific rule matched
    if not recommendations:
        recommendations.append({
            'Priority': 'Low',
            'Action': 'General Retention Outreach',
            'Detail': 'Schedule a customer satisfaction call and offer loyalty rewards.',
            'Expected Impact': 'General retention improvement ~8%'
        })

    return recommendations


def get_revenue_at_risk(churn_probability, cltv):
    """Calculate estimated revenue at risk for a customer."""
    return round(churn_probability * cltv, 2)