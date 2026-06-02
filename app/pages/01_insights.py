import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import importlib.util
import sys
import os

# dynamic root path - pages are 2 levels deep inside app/pages/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

# loading shared styles
spec = importlib.util.spec_from_file_location("styles",
       os.path.join(ROOT, "app", "styles.py"))
styles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(styles)

st.set_page_config(page_title="Churn Insights", page_icon="📊", layout="wide")
styles.apply_styles()

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(ROOT, 'data/processed/telco_engineered.csv'))

df = load_data()

# sidebar filters
st.sidebar.markdown("<hr style='border-color: #00B4D8;'/>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#00B4D8; font-weight:bold;'>🔧 Filter Dataset</p>",
                    unsafe_allow_html=True)

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=['Month-to-month', 'One year', 'Two year'],
    default=['Month-to-month', 'One year', 'Two year']
)
tenure_filter = st.sidebar.multiselect(
    "Tenure Group",
    options=['New', 'Developing', 'Established', 'Loyal'],
    default=['New', 'Developing', 'Established', 'Loyal']
)
value_filter = st.sidebar.multiselect(
    "Value Segment",
    options=['Low Value', 'Medium Value', 'High Value'],
    default=['Low Value', 'Medium Value', 'High Value']
)

contract_map = {
    'Month-to-month': 'Contract_Month-to-month',
    'One year': 'Contract_One year',
    'Two year': 'Contract_Two year'
}
tenure_map = {
    'New': 'Tenure Group_New', 'Developing': 'Tenure Group_Developing',
    'Established': 'Tenure Group_Established', 'Loyal': 'Tenure Group_Loyal'
}
value_map = {
    'Low Value': 'Value Segment_Low Value',
    'Medium Value': 'Value Segment_Medium Value',
    'High Value': 'Value Segment_High Value'
}

contract_cols = [contract_map[c] for c in contract_filter]
tenure_cols_filter = [tenure_map[t] for t in tenure_filter]
value_cols_filter = [value_map[v] for v in value_filter]

mask = pd.Series([True] * len(df), index=df.index)
if contract_cols:
    mask = mask & df[contract_cols].any(axis=1)
if tenure_cols_filter:
    mask = mask & df[tenure_cols_filter].any(axis=1)
if value_cols_filter:
    mask = mask & df[value_cols_filter].any(axis=1)
df = df[mask]

st.sidebar.markdown(f"<p style='color:#A0A0B0; font-size:12px;'>Showing {len(df):,} customers</p>",
                    unsafe_allow_html=True)

st.markdown("<h1>📊 Churn Insights</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#A0A0B0;'>Explore churn patterns across the dataset.</p>",
            unsafe_allow_html=True)
st.markdown("---")

total = len(df)
churned = df['Churn Value'].sum()
churn_rate = churned / total * 100
revenue_at_risk = df[df['Churn Value'] == 1]['CLTV'].sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Customers", f"{total:,}")
m2.metric("Churned Customers", f"{churned:,}")
m3.metric("Churn Rate", f"{churn_rate:.1f}%")
m4.metric("Total CLTV at Risk", f"${revenue_at_risk:,.0f}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    contract_data = df.groupby(
        df[['Contract_Month-to-month', 'Contract_One year', 'Contract_Two year']]
        .idxmax(axis=1))['Churn Value'].mean().mul(100).round(1).reset_index()
    contract_data.columns = ['Contract', 'Churn Rate %']
    contract_data['Contract'] = contract_data['Contract'].str.replace('Contract_', '')
    fig = px.bar(contract_data, x='Contract', y='Churn Rate %',
                 color='Churn Rate %', color_continuous_scale='Reds',
                 title='Churn Rate by Contract Type')
    fig.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                      font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    tenure_cols = ['Tenure Group_New', 'Tenure Group_Developing',
                   'Tenure Group_Established', 'Tenure Group_Loyal']
    tenure_churn = []
    for col in tenure_cols:
        rate = df[df[col] == 1]['Churn Value'].mean() * 100
        tenure_churn.append({'Tenure Group': col.replace('Tenure Group_', ''),
                             'Churn Rate %': round(rate, 1)})
    tenure_df = pd.DataFrame(tenure_churn)
    fig2 = px.bar(tenure_df, x='Tenure Group', y='Churn Rate %',
                  color='Churn Rate %', color_continuous_scale='Blues',
                  title='Churn Rate by Tenure Group')
    fig2.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                       font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                       showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    value_cols = ['Value Segment_High Value', 'Value Segment_Low Value',
                  'Value Segment_Medium Value']
    value_churn = []
    for col in value_cols:
        rate = df[df[col] == 1]['Churn Value'].mean() * 100
        value_churn.append({'Value Segment': col.replace('Value Segment_', ''),
                            'Churn Rate %': round(rate, 1)})
    value_df = pd.DataFrame(value_churn)
    fig3 = px.bar(value_df, x='Value Segment', y='Churn Rate %',
                  color='Churn Rate %', color_continuous_scale='Oranges',
                  title='Churn Rate by Customer Value Segment')
    fig3.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                       font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                       showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.histogram(df, x='Monthly Charges',
                        color=df['Churn Value'].map({0: 'No Churn', 1: 'Churn'}),
                        barmode='overlay', title='Monthly Charges Distribution by Churn',
                        color_discrete_map={'No Churn': '#00B4D8', 'Churn': '#FF4B4B'})
    fig4.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                       font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                       legend=dict(font=dict(color='#E0E0E0')))
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

col5, col6 = st.columns(2)

with col5:
    senior_churn = df.groupby('Senior Citizen')['Churn Value'].mean().mul(100).round(1).reset_index()
    senior_churn['Senior Citizen'] = senior_churn['Senior Citizen'].map({0: 'Non-Senior', 1: 'Senior'})
    senior_churn.columns = ['Senior Citizen', 'Churn Rate %']
    fig6 = px.bar(senior_churn, x='Senior Citizen', y='Churn Rate %',
                  color='Churn Rate %', color_continuous_scale='Reds',
                  title='Churn Rate by Senior Citizen')
    fig6.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                       font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                       showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

with col6:
    dep_churn = df.groupby('Dependents')['Churn Value'].mean().mul(100).round(1).reset_index()
    dep_churn['Dependents'] = dep_churn['Dependents'].map({0: 'No Dependents', 1: 'Has Dependents'})
    dep_churn.columns = ['Dependents', 'Churn Rate %']
    fig7 = px.bar(dep_churn, x='Dependents', y='Churn Rate %',
                  color='Churn Rate %', color_continuous_scale='Blues',
                  title='Churn Rate by Dependents')
    fig7.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                       font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                       showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")

@st.cache_data
def load_churn_reasons():
    return pd.read_csv(os.path.join(ROOT, 'data/processed/churn_reasons.csv'))

reasons_df = load_churn_reasons()
reasons_df = reasons_df[reasons_df['Churn Value'] == 1]
reasons_count = reasons_df['Churn Reason'].value_counts().reset_index()
reasons_count.columns = ['Churn Reason', 'Count']

st.markdown("<h3>Top Reasons Customers Churned</h3>", unsafe_allow_html=True)
fig8 = px.bar(reasons_count, x='Count', y='Churn Reason', orientation='h',
              color='Count', color_continuous_scale='Reds',
              title='Why Customers Churned')
fig8.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                   font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                   showlegend=False, height=500)
st.plotly_chart(fig8, use_container_width=True)

st.markdown("---")

@st.cache_resource
def load_shap():
    shap_values = np.load(os.path.join(ROOT, 'src/models/shap_values.npy'))
    with open(os.path.join(ROOT, 'src/models/feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    return shap_values, feature_names

shap_values, feature_names = load_shap()
mean_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': mean_shap
}).sort_values('Importance', ascending=True).tail(15)

st.markdown("<h3>Top Churn Drivers (SHAP Feature Importance)</h3>", unsafe_allow_html=True)
fig5 = px.bar(shap_df, x='Importance', y='Feature', orientation='h',
              color='Importance', color_continuous_scale='Blues',
              title='Top 15 Features by SHAP Importance')
fig5.update_layout(paper_bgcolor='#2A2A3E', plot_bgcolor='#2A2A3E',
                   font=dict(color='#E0E0E0'), title_font=dict(color='#00B4D8'),
                   showlegend=False, height=500)
st.plotly_chart(fig5, use_container_width=True)