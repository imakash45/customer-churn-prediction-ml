import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
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
from src.utils.helpers import get_risk_level, format_currency, get_churn_summary

st.set_page_config(page_title="Churn Prediction", page_icon="🔍", layout="wide")
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

st.markdown("<h1>🔍 Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#A0A0B0;'>Enter customer details below to predict churn probability.</p>",
            unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3>Demographics</h3>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.markdown("<h3>Account Info</h3>", unsafe_allow_html=True)
    tenure = st.slider("Tenure Months", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                     float(monthly_charges * tenure))
    cltv = st.number_input("Customer Lifetime Value ($)", 0, 10000, 4400)

with col3:
    st.markdown("<h3>Services</h3>", unsafe_allow_html=True)
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes"])
    backup = st.selectbox("Online Backup", ["No", "Yes"])
    device = st.selectbox("Device Protection", ["No", "Yes"])
    tech = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

st.markdown("---")

if st.button("🔍 Predict Churn", use_container_width=True):
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
    summary = get_churn_summary(churn_prob, cltv)

    st.markdown("<h2>📊 Prediction Results</h2>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric("Churn Probability", f"{churn_prob:.1%}")
    r2.metric("Risk Level", risk_level)
    r3.metric("Revenue at Risk", format_currency(churn_prob * cltv))

    st.markdown(f"""
    <div class='card'>
        <p style='color:#E0E0E0;'>{summary}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3>🔎 Why is this customer at risk?</h3>", unsafe_allow_html=True)

    shap_vals = explainer.shap_values(scaled)
    if len(np.array(shap_vals).shape) == 3:
        shap_row = shap_vals[0, :, 1]
    else:
        shap_row = shap_vals[1][0]

    top_drivers = get_top_drivers(shap_row, feature_names, top_n=5)

    fig = px.bar(top_drivers, x='SHAP Value', y='Feature', orientation='h',
                 color='SHAP Value', color_continuous_scale='Reds',
                 title='Top 5 Churn Drivers for this Customer')
    fig.update_layout(showlegend=False, height=350,
                      paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                      font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'))
    st.plotly_chart(fig, use_container_width=True)