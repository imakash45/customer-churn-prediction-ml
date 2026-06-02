import streamlit as st

def apply_styles(current_page="home"):
    st.markdown("""
    <style>
    [data-testid="stHeader"] {
        background-color: #1E1E2E;
        border-bottom: none;
    }
    .stApp {
        background-color: #1E1E2E;
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] {
        background-color: #16213E;
        border-right: 1px solid #00B4D8;
    }
    [data-testid="stSidebar"] * {
        color: #E0E0E0 !important;
    }
    .card {
        background-color: #2A2A3E;
        border: 1px solid #00B4D8;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    [data-testid="stMetric"] {
        background-color: #2A2A3E;
        border: 1px solid #00B4D8;
        border-radius: 8px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #00B4D8 !important;
        font-size: 14px !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 28px !important;
    }
    .stButton > button {
        background-color: #00B4D8;
        color: #1E1E2E;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #0096B7;
        color: white;
    }
    h1, h2, h3 {
        color: #00B4D8 !important;
    }
    hr {
        border-color: #00B4D8;
        opacity: 0.3;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    label, .stSelectbox label, .stSlider label,
    .stNumberInput label, .stTextInput label {
        color: #E0E0E0 !important;
        font-size: 14px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #2A2A3E !important;
        color: #E0E0E0 !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: #E0E0E0 !important;
    }
    .stSlider p {
        color: #E0E0E0 !important;
    }
    .stNumberInput input {
        background-color: #2A2A3E !important;
        color: #E0E0E0 !important;
    }
    .stMarkdown p {
        color: #E0E0E0 !important;
    }
    [data-testid="stToolbar"] {
        visibility: hidden;
    }
    [data-testid="collapsedControl"] {
        background-color: #FFFFFF !important;
        border-radius: 0 5px 5px 0 !important;
        width: 25px !important;
        height: 50px !important;
        color: #1E1E2E !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

    # sidebar branding
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h2 style='color: #00B4D8;'>📡 Churn Intelligence</h2>
        <p style='color: #A0A0B0; font-size: 12px;'>Telecom Customer Retention System</p>
        <hr style='border-color: #00B4D8; opacity:0.3;'/>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <p style='color: #A0A0B0; font-size: 12px; padding: 10px;'>
        📌 Model: Random Forest<br>
        📌 AUC Score: 0.85<br>
        📌 Dataset: 7043 customers<br>
        📌 Features: 44 engineered
    </p>
    """, unsafe_allow_html=True)