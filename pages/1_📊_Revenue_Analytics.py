import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data
from utils.kpi_calculator import revenue_by_period, revenue_by_dimension, pareto_analysis
from utils.visualizations import plot_trend, plot_bar, plot_pie, plot_heatmap

st.set_page_config(page_title="Revenue Analytics", page_icon="📊", layout="wide")
st.title("📊 Revenue Analytics")

data = load_data()
sales = data['sales'].merge(data['products'][['product_id', 'category', 'brand']], on='product_id')

# Filters
with st.expander("🎛️ Filters", expanded=True):
    c1, c2, c3 = st.columns(3)
    period = c1.selectbox("Periodicity", ['D', 'W', 'M', 'Q', 'Y'], index=2)
    region_filter = c2.multiselect("Region", sales['region'].unique(), default=sales['region'].unique())
    channel_filter = c3.multiselect("Channel", sales['channel'].unique(), default=sales['channel'].unique())

sales_f = sales[sales['region'].isin(region_filter) & sales['channel'].isin(channel_filter)]

# KPIs
st.subheader("🎯 Key Metrics")
k1, k2, k3, k4, k5 = st.columns(5)
rev = sales_f['net_amount'].sum()
qty = sales_f['quantity'].sum()
aov = rev / sales_f['order_id'].nunique()
skus = sales_f['product_id'].nunique()
outlets = sales_f['outlet_id'].nunique()

k1.metric("Revenue", f"₹{rev/1e7:.2f} Cr")
k2.metric("Volume", f"{qty/1e6:.2f}M units")
k3.metric("Avg Order Value", f"₹{aov:,.0f}")
k4.metric("SKUs Sold", skus)
k5.metric("Active Outlets", outlets)

# Trend
st.subheader("📈 Revenue Trend")
trend = revenue_by_period(sales_f, period)
trend.columns = ['Period', 'Revenue']
st.plotly_chart(plot_trend(trend, 'Period', 'Revenue', f'{period}ly Revenue Trend'), use_container_width=True)

# Channel & Region
c1, c2 = st.columns(2)
with c1:
    st.subheader("🏪 Channel Mix")
    channel = revenue_by_dimension(sales_f, 'channel')
    st.plotly_chart(plot_pie(channel, 'channel', 'net_amount', 'Revenue by Channel'), use_container_width=True)
with c2:
    st.subheader("🌍 Region Mix")
    region = revenue_by_dimension(sales_f, 'region')
    st.plotly_chart(plot_bar(region, 'region', 'net_amount', 'Revenue by Region', color_col='region'), use_container_width=True)

# Heatmap
st.subheader("🔥 Region × Channel Heatmap")
st.plotly_chart(plot_heatmap(sales_f, 'channel', 'region', 'net_amount', 'Revenue Heatmap'), use_container_width=True)

# Pareto
st.subheader("📊 Top SKUs - 80/20 Pareto")
top_n = st.slider("Show Top N SKUs", 10, 100, 30)
pareto = pareto_analysis(sales_f, 'product_id').head(top_n)
pareto = pareto.merge(data['products'][['product_id', 'product_name', 'category']], on='product_id')

fig = px.bar(pareto, x='product_name', y='net_amount', 
             hover_data=['cum_pct'], title=f'Top {top_n} SKUs Revenue Contribution')
fig.add_scatter(x=pareto['product_name'], y=pareto['cum_pct']*rev/100, 
                mode='lines+markers', name='Cumulative %', yaxis='y2')
fig.update_layout(yaxis2=dict(title='Cumulative %', overlaying='y', side='right'))
st.plotly_chart(fig, use_container_width=True)

# Category / Brand
c1, c2 = st.columns(2)
with c1:
    cat = revenue_by_dimension(sales_f, 'category')
    st.plotly_chart(plot_bar(cat.head(10), 'category', 'net_amount', 'Top Categories', color_col='category'), use_container_width=True)
with c2:
    brand = revenue_by_dimension(sales_f, 'brand')
    st.plotly_chart(plot_bar(brand, 'brand', 'net_amount', 'Brand Performance', color_col='brand'), use_container_width=True)
