import streamlit as st
import importlib.util
import sys
import os

# dynamic root path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# loading shared styles
spec = importlib.util.spec_from_file_location("styles",
       os.path.join(ROOT, "app", "styles.py"))
styles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(styles)

st.set_page_config(
    page_title="Churn Intelligence System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

styles.apply_styles()

st.markdown("<h1 style='text-align:center;'>📡 Customer Churn Intelligence System</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#A0A0B0;'>Predict churn · Understand risk · Retain customers</p>",
            unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='card'>
        <h3>🔍 Prediction</h3>
        <p style='color:#A0A0B0;'>Enter customer details and get instant churn probability with explanation.</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='card'>
        <h3>📊 Insights</h3>
        <p style='color:#A0A0B0;'>Explore churn patterns across contract type, tenure, services and demographics.</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='card'>
        <h3>🎯 Retention</h3>
        <p style='color:#A0A0B0;'>Get data-driven retention recommendations based on individual churn drivers.</p>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='card'>
        <h3>💰 Risk Dashboard</h3>
        <p style='color:#A0A0B0;'>View revenue at risk, customer segments and high risk customer list.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class='card'>
    <h3>About This Project</h3>
    <p style='color:#A0A0B0;'>
        This end-to-end machine learning system identifies telecom customers likely to cancel
        their subscription and recommends actionable strategies to retain them before they leave.
        Built using Random Forest with SHAP explainability for transparent, business-focused predictions.
    </p>
</div>
""", unsafe_allow_html=True)