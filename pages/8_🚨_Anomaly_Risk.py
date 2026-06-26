import streamlit as st
from sklearn.ensemble import IsolationForest
import plotly.express as px
import pandas as pd
from utils.data_loader import load_data

st.set_page_config(page_title="Anomaly & Risk", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly & Risk Detection")

data = load_data()

tab1, tab2, tab3 = st.tabs(["📊 Sales Anomalies", "💰 Bad Debt Risk", "📦 Inventory Alerts"])

with tab1:
    st.subheader("Outlier Transactions")
    sales = data['sales'].copy()
    daily = sales.groupby('order_date')['net_amount'].sum().reset_index()
    
    model = IsolationForest(contamination=0.05, random_state=42)
    daily['anomaly'] = model.fit_predict(daily[['net_amount']])
    
    anomalies = daily[daily['anomaly'] == -1]
    fig = px.scatter(daily, x='order_date', y='net_amount', 
                     color=daily['anomaly'].map({1:'Normal', -1:'Anomaly'}),
                     color_discrete_map={'Normal':'blue', 'Anomaly':'red'},
                     title='Daily Revenue Anomaly Detection')
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"⚠️ Detected {len(anomalies)} anomalous days")
    st.dataframe(anomalies, use_container_width=True)

with tab2:
    st.subheader("High Risk Receivables")
    ar = data['receivables'].merge(data['outlets'][['outlet_id','outlet_name','credit_limit']], on='outlet_id')
    ar['risk_score'] = (ar['aging_days'] / 90).clip(0, 1) * 100
    
    high_risk = ar[ar['risk_score'] > 70].sort_values('risk_score', ascending=False)
    st.metric("High Risk Invoices", len(high_risk))
    st.metric("High Risk Value", f"₹{high_risk['outstanding'].sum()/1e5:.2f} L")
    st.dataframe(high_risk[['outlet_name', 'invoice_amount', 'outstanding', 'aging_days', 'risk_score']].head(50), use_container_width=True)

with tab3:
    st.subheader("Low Stock Alerts")
    inv = data['inventory']
    low_stock = inv[inv['closing_stock'] < inv['opening_stock'] * 0.2]
    st.metric("Low Stock Instances", len(low_stock))
    st.dataframe(low_stock.head(50), use_container_width=True)
