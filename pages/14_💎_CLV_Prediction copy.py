import streamlit as st
import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import plotly.express as px
from utils.data_loader import load_data

st.set_page_config(page_title="CLV Prediction", page_icon="💎", layout="wide")
st.title("💎 Customer Lifetime Value Prediction")
st.markdown("### BG/NBD + Gamma-Gamma Model for CLV")

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
except ImportError:
    st.error("Install lifetimes: `pip install lifetimes`")
    st.stop()

data = load_data()
sales = data['sales'].copy()
sales = sales.merge(data['outlets'][['outlet_id','outlet_name']], on='outlet_id')

# ============================================
# RFM SUMMARY
# ============================================
observation_period_end = sales['order_date'].max()
rfm = summary_data_from_transaction_data(
    sales, 'outlet_id', 'order_date', 'net_amount',
    observation_period_end=observation_period_end
)

st.subheader("📊 RFM Summary Statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", len(rfm))
c2.metric("Avg Frequency", f"{rfm['frequency'].mean():.2f}")
c3.metric("Avg Recency (days)", f"{rfm['recency'].mean():.0f}")
c4.metric("Avg Monetary (₹)", f"₹{rfm['monetary_value'].mean():,.0f}")

# ============================================
# BG/NBD MODEL (Purchase Frequency)
# ============================================
st.subheader("🎯 BG/NBD Model - Future Purchases")

returning = rfm[rfm['frequency'] > 0]

with st.spinner("Training BG/NBD..."):
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(returning['frequency'], returning['recency'], returning['T'])

# Predict future purchases (next 90, 180, 365 days)
periods = {'90 days': 90, '180 days': 180, '365 days': 365}
predictions = pd.DataFrame({
    'Customer': returning.index,
    'Predicted_Purchases_90d': bgf.conditional_expected_number_of_purchases_up_to_time(90, returning['frequency'], returning['recency'], returning['T']),
    'Predicted_Purchases_180d': bgf.conditional_expected_number_of_purchases_up_to_time(180, returning['frequency'], returning['recency'], returning['T']),
    'Predicted_Purchases_365d': bgf.conditional_expected_number_of_purchases_up_to_time(365, returning['frequency'], returning['recency'], returning['T']),
})

predictions = predictions.merge(rfm, left_on='Customer', right_index=True)
predictions = predictions.merge(data['outlets'][['outlet_id','outlet_name']], 
                                  left_on='Customer', right_on='outlet_id')

st.dataframe(predictions[['outlet_name','frequency','monetary_value',
                           'Predicted_Purchases_90d','Predicted_Purchases_365d']].head(20), 
             use_container_width=True)

# ============================================
# GAMMA-GAMMA MODEL (Monetary Value)
# ============================================
st.subheader("💰 Gamma-Gamma Model - Customer Value")

returning_filtered = returning[returning['monetary_value'] > 0]

with st.spinner("Training Gamma-Gamma..."):
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(returning_filtered['frequency'], returning_filtered['monetary_value'])

# Calculate CLV (12 months)
returning_filtered['predicted_avg_value'] = ggf.conditional_expected_average_profit(
    returning_filtered['frequency'], returning_filtered['monetary_value']
)

returning_filtered['CLV_12m'] = ggf.customer_lifetime_value(
    bgf, returning_filtered['frequency'], returning_filtered['recency'], 
    returning_filtered['T'], returning_filtered['monetary_value'],
    time=12, discount_rate=0.01
)

returning_filtered = returning_filtered.merge(data['outlets'][['outlet_id','outlet_name']], 
                                                left_index=True, right_on='outlet_id')

# Merge channel and region from sales data
outlet_channels = sales[['outlet_id', 'channel', 'region']].drop_duplicates()
returning_filtered = returning_filtered.merge(outlet_channels, on='outlet_id', how='left')

# CLV Segments
returning_filtered['CLV_Segment'] = pd.qcut(returning_filtered['CLV_12m'], 
                                              q=4, 
                                              labels=['🥉 Bronze', '🥈 Silver', '🥇 Gold', '💎 Platinum'])

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg CLV (12m)", f"₹{returning_filtered['CLV_12m'].mean():,.0f}")
c2.metric("Median CLV (12m)", f"₹{returning_filtered['CLV_12m'].median():,.0f}")
c3.metric("Top 10% CLV", f"₹{returning_filtered['CLV_12m'].quantile(0.9):,.0f}")
c4.metric("Total CLV (12m)", f"₹{returning_filtered['CLV_12m'].sum()/1e7:.2f} Cr")

# ============================================
# CLV VISUALIZATIONS
# ============================================
st.subheader("📈 CLV Distribution")

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(returning_filtered, x='CLV_12m', nbins=50, 
                       color='CLV_Segment', title='CLV Distribution by Segment')
    st.plotly_chart(fig, use_container_width=True)

with c2:
    seg_summary = returning_filtered.groupby('CLV_Segment').agg(
        count=('outlet_id','count'),
        avg_clv=('CLV_12m','mean'),
        total_clv=('CLV_12m','sum')
    ).reset_index()
    
    fig = px.bar(seg_summary, x='CLV_Segment', y='total_clv', color='CLV_Segment',
                 title='Total CLV Contribution by Segment')
    st.plotly_chart(fig, use_container_width=True)

# Top customers
st.subheader("🏆 Top 20 Customers by CLV")
top_clv = returning_filtered.nlargest(20, 'CLV_12m')[['outlet_name','channel','region','frequency','monetary_value','CLV_12m']]
st.dataframe(top_clv.style.format({'monetary_value':'₹{:,.0f}', 'CLV_12m':'₹{:,.0f}'}), use_container_width=True)

# Channel-wise CLV
st.subheader("🏪 Channel-wise Average CLV")
channel_clv = returning_filtered.groupby('channel')['CLV_12m'].agg(['mean','sum','count']).reset_index()
channel_clv.columns = ['Channel', 'Avg CLV', 'Total CLV', 'Customer Count']

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(channel_clv, x='Channel', y='Avg CLV', color='Channel',
                 title='Average CLV by Channel')
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.bar(channel_clv, x='Channel', y='Total CLV', color='Channel',
                 title='Total CLV Contribution by Channel')
    st.plotly_chart(fig, use_container_width=True)
