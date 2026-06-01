import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import importlib.util
import sys

sys.path.append('E:\\Churn-Intelligence')

# loading shared styles
spec = importlib.util.spec_from_file_location("styles", "E:\\Churn-Intelligence\\app\\styles.py")
styles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(styles)

from src.utils.helpers import get_risk_level, format_currency

st.set_page_config(page_title="Risk Dashboard", page_icon="💰", layout="wide")
styles.apply_styles()

# loading data and predictions once
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/telco_engineered.csv')
    probs = pd.read_csv('data/processed/churn_probabilities.csv')
    df['Churn Probability'] = probs['Churn Probability']
    df['Revenue at Risk'] = df['Churn Probability'] * df['CLTV']
    df['Risk Level'] = df['Churn Probability'].apply(lambda x: get_risk_level(x)[0])
    return df

df = load_data()

# sidebar filters
st.sidebar.markdown("<hr style='border-color: #00B4D8;'/>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#00B4D8; font-weight:bold;'>🔧 Filter Customers</p>",
                    unsafe_allow_html=True)

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=['High Risk', 'Medium Risk', 'Low Risk'],
    default=['High Risk', 'Medium Risk', 'Low Risk']
)

value_segment_filter = st.sidebar.multiselect(
    "Value Segment",
    options=['High Value', 'Medium Value', 'Low Value'],
    default=['High Value', 'Medium Value', 'Low Value']
)

tenure_segment_filter = st.sidebar.multiselect(
    "Tenure Group",
    options=['New', 'Developing', 'Established', 'Loyal'],
    default=['New', 'Developing', 'Established', 'Loyal']
)

# applying filters
value_map = {
    'High Value': 'Value Segment_High Value',
    'Medium Value': 'Value Segment_Medium Value',
    'Low Value': 'Value Segment_Low Value'
}
tenure_map = {
    'New': 'Tenure Group_New', 'Developing': 'Tenure Group_Developing',
    'Established': 'Tenure Group_Established', 'Loyal': 'Tenure Group_Loyal'
}

mask = pd.Series([True] * len(df), index=df.index)

if risk_filter:
    mask = mask & df['Risk Level'].isin(risk_filter)
if value_segment_filter:
    value_cols_f = [value_map[v] for v in value_segment_filter]
    mask = mask & df[value_cols_f].any(axis=1)
if tenure_segment_filter:
    tenure_cols_f = [tenure_map[t] for t in tenure_segment_filter]
    mask = mask & df[tenure_cols_f].any(axis=1)

df = df[mask]

st.sidebar.markdown(f"<p style='color:#A0A0B0; font-size:12px;'>Showing {len(df):,} customers</p>",
                    unsafe_allow_html=True)

st.markdown("<h1>💰 Revenue Risk Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#A0A0B0;'>Portfolio-level view of churn risk and revenue impact.</p>",
            unsafe_allow_html=True)
st.markdown("---")

# top level metrics
total_customers = len(df)
high_risk = len(df[df['Risk Level'] == 'High Risk'])
medium_risk = len(df[df['Risk Level'] == 'Medium Risk'])
low_risk = len(df[df['Risk Level'] == 'Low Risk'])
total_revenue_at_risk = df['Revenue at Risk'].sum()
high_risk_revenue = df[df['Risk Level'] == 'High Risk']['Revenue at Risk'].sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Customers", f"{total_customers:,}")
m2.metric("High Risk", f"{high_risk:,}", delta=f"{high_risk/total_customers*100:.1f}%")
m3.metric("Medium Risk", f"{medium_risk:,}", delta=f"{medium_risk/total_customers*100:.1f}%")
m4.metric("Total Revenue at Risk", format_currency(total_revenue_at_risk))
m5.metric("High Risk Revenue", format_currency(high_risk_revenue))

st.markdown("---")

# risk distribution and revenue at risk by segment
col1, col2 = st.columns(2)

with col1:
    # risk level distribution
    risk_counts = df['Risk Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    color_map = {'High Risk': '#FF4B4B', 'Medium Risk': '#FFA500', 'Low Risk': '#00CC44'}

    fig1 = px.pie(risk_counts, values='Count', names='Risk Level',
                  title='Customer Risk Distribution',
                  color='Risk Level', color_discrete_map=color_map)
    fig1.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                       font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                       legend=dict(font=dict(color='#E0E0E0')))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # revenue at risk by risk level
    rev_risk = df.groupby('Risk Level')['Revenue at Risk'].sum().reset_index()
    rev_risk.columns = ['Risk Level', 'Revenue at Risk']

    fig2 = px.bar(rev_risk, x='Risk Level', y='Revenue at Risk',
                  color='Risk Level', color_discrete_map=color_map,
                  title='Revenue at Risk by Risk Level')
    fig2.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                       font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                       showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# churn probability distribution and revenue at risk by value segment
col3, col4 = st.columns(2)

with col3:
    fig3 = px.histogram(df, x='Churn Probability', nbins=50,
                        color='Risk Level', color_discrete_map=color_map,
                        title='Churn Probability Distribution')
    fig3.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                       font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                       legend=dict(font=dict(color='#E0E0E0')))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # revenue at risk by value segment
    value_cols = ['Value Segment_High Value', 'Value Segment_Low Value',
                  'Value Segment_Medium Value']
    value_rev = []
    for col in value_cols:
        segment = col.replace('Value Segment_', '')
        rev = df[df[col] == 1]['Revenue at Risk'].sum()
        value_rev.append({'Value Segment': segment, 'Revenue at Risk': rev})
    value_rev_df = pd.DataFrame(value_rev)

    fig4 = px.bar(value_rev_df, x='Value Segment', y='Revenue at Risk',
                  color='Revenue at Risk', color_continuous_scale='Reds',
                  title='Revenue at Risk by Value Segment')
    fig4.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                       font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                       showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# high risk customers table
st.markdown("<h3>🚨 Top 20 High Risk Customers</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#A0A0B0;'>Customers with highest churn probability and revenue at risk.</p>",
            unsafe_allow_html=True)

# loading original data for customer index
top20 = df.nlargest(20, 'Churn Probability')[
    ['Churn Probability', 'Risk Level', 'CLTV',
     'Revenue at Risk', 'Tenure Months', 'Monthly Charges']
].reset_index()

top20.columns = ['Customer Index', 'Churn Probability', 'Risk Level',
                 'CLTV', 'Revenue at Risk', 'Tenure Months', 'Monthly Charges']

top20['Churn Probability'] = top20['Churn Probability'].apply(lambda x: f"{x:.1%}")
top20['Revenue at Risk'] = top20['Revenue at Risk'].apply(lambda x: format_currency(x))
top20['CLTV'] = top20['CLTV'].apply(lambda x: format_currency(x))

st.dataframe(top20, use_container_width=True, height=400)

st.markdown("---")

# churn probability vs cltv scatter
st.markdown("<h3>📍 Churn Probability vs Customer Lifetime Value</h3>", unsafe_allow_html=True)

fig5 = px.scatter(df, x='Churn Probability', y='CLTV',
                  color='Risk Level', color_discrete_map=color_map,
                  size='Revenue at Risk', opacity=0.6,
                  title='Churn Probability vs CLTV — Bubble size = Revenue at Risk',
                  hover_data=['Tenure Months', 'Monthly Charges'])
fig5.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                   font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                   legend=dict(font=dict(color='#E0E0E0')))
st.plotly_chart(fig5, use_container_width=True)