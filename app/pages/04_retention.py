import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import importlib.util
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

spec = importlib.util.spec_from_file_location("styles",
       os.path.join(ROOT, "app", "styles.py"))
styles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(styles)

from src.preprocessing.pipeline import preprocess_input, align_features
from src.explainability.explainer import get_top_drivers
from src.retention.recommender import get_retention_recommendations, get_revenue_at_risk
from src.utils.helpers import get_risk_level, format_currency

st.set_page_config(page_title="Retention Engine", page_icon="🎯", layout="wide")
styles.apply_styles()

@st.cache_resource
def load_models():
    model = joblib.load(os.path.join(ROOT, 'src/models/final_model.pkl'))
    scaler = joblib.load(os.path.join(ROOT, 'src/models/scaler.pkl'))
    explainer = joblib.load(os.path.join(ROOT, 'src/models/shap_explainer.pkl'))
    with open(os.path.join(ROOT, 'src/models/feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    return model, scaler, explainer, feature_names

model, scaler, explainer, feature_names = load_models()

st.markdown("<h1>🎯 Retention Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#A0A0B0;'>Enter customer details to get personalized retention strategies.</p>",
            unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3>Demographics</h3>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"], key="ret_gender")
    senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="ret_senior")
    partner = st.selectbox("Partner", ["No", "Yes"], key="ret_partner")
    dependents = st.selectbox("Dependents", ["No", "Yes"], key="ret_dependents")

with col2:
    st.markdown("<h3>Account Info</h3>", unsafe_allow_html=True)
    tenure = st.slider("Tenure Months", 0, 72, 12, key="ret_tenure")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="ret_contract")
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ], key="ret_payment")
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"], key="ret_paperless")
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, key="ret_monthly")
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                     float(monthly_charges * tenure), key="ret_total")
    cltv = st.number_input("Customer Lifetime Value ($)", 0, 10000, 4400, key="ret_cltv")

with col3:
    st.markdown("<h3>Services</h3>", unsafe_allow_html=True)
    phone = st.selectbox("Phone Service", ["No", "Yes"], key="ret_phone")
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"], key="ret_multiple")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="ret_internet")
    security = st.selectbox("Online Security", ["No", "Yes"], key="ret_security")
    backup = st.selectbox("Online Backup", ["No", "Yes"], key="ret_backup")
    device = st.selectbox("Device Protection", ["No", "Yes"], key="ret_device")
    tech = st.selectbox("Tech Support", ["No", "Yes"], key="ret_tech")
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], key="ret_tv")
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], key="ret_movies")

st.markdown("---")

if st.button("🎯 Get Retention Recommendations", use_container_width=True):
    customer = {
        'Gender': gender, 'Senior Citizen': senior,
        'Partner': partner, 'Dependents': dependents,
        'Tenure Months': tenure, 'Phone Service': phone,
        'Multiple Lines': multiple, 'Internet Service': internet,
        'Online Security': security, 'Online Backup': backup,
        'Device Protection': device, 'Tech Support': tech,
        'Streaming TV': streaming_tv, 'Streaming Movies': streaming_movies,
        'Contract': contract, 'Paperless Billing': paperless,
        'Payment Method': payment, 'Monthly Charges': monthly_charges,
        'Total Charges': total_charges, 'CLTV': cltv
    }

    processed = preprocess_input(customer)
    aligned = align_features(processed, feature_names)
    scaled = scaler.transform(aligned)

    churn_prob = model.predict_proba(scaled)[0][1]
    risk_level, risk_color = get_risk_level(churn_prob)
    revenue_at_risk = get_revenue_at_risk(churn_prob, cltv)

    shap_vals = explainer.shap_values(scaled)
    if len(np.array(shap_vals).shape) == 3:
        shap_row = shap_vals[0, :, 1]
    else:
        shap_row = shap_vals[1][0]

    top_drivers = get_top_drivers(shap_row, feature_names, top_n=5)

    st.markdown("<h2>📊 Customer Risk Summary</h2>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric("Churn Probability", f"{churn_prob:.1%}")
    r2.metric("Risk Level", risk_level)
    r3.metric("Revenue at Risk", format_currency(revenue_at_risk))

    st.markdown("---")
    st.markdown("<h3>🔎 Key Churn Drivers</h3>", unsafe_allow_html=True)
    for _, row in top_drivers.iterrows():
        st.markdown(f"""
        <div class='card' style='margin:5px 0; padding:10px;'>
            <span style='color:#00B4D8; font-weight:bold;'>{row['Feature']}</span>
            <span style='color:#A0A0B0; float:right;'>Impact: {row['SHAP Value']:.4f}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    recommendations = get_retention_recommendations(customer, top_drivers)
    st.markdown("<h3>🎯 Recommended Retention Actions</h3>", unsafe_allow_html=True)

    priority_colors = {'High': '#FF4B4B', 'Medium': '#FFA500', 'Low': '#00CC44'}
    for i, rec in enumerate(recommendations):
        color = priority_colors.get(rec['Priority'], '#00B4D8')
        st.markdown(f"""
        <div class='card' style='border-left: 4px solid {color}; margin: 10px 0;'>
            <div style='display:flex; justify-content:space-between;'>
                <h4 style='color:{color}; margin:0;'>{i+1}. {rec['Action']}</h4>
                <span style='color:{color}; font-weight:bold;'>{rec['Priority']} Priority</span>
            </div>
            <p style='color:#E0E0E0; margin:8px 0;'>{rec['Detail']}</p>
            <p style='color:#A0A0B0; margin:0;'>📈 {rec['Expected Impact']}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div class='card' style='border: 1px solid #FFA500;'>
        <h3 style='color:#FFA500;'>💰 Revenue Impact</h3>
        <p style='color:#E0E0E0;'>
            If this customer churns, estimated revenue loss is
            <span style='color:#FF4B4B; font-weight:bold;'>{format_currency(revenue_at_risk)}</span>.
            Implementing the above retention actions can significantly reduce this risk.
        </p>
    </div>""", unsafe_allow_html=True)