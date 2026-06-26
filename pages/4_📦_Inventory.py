import streamlit as st
import plotly.express as px
from utils.data_loader import load_data

st.set_page_config(page_title="Inventory", page_icon="📦", layout="wide")
st.title("📦 Inventory Health")

data = load_data()
inv = data['inventory'].merge(data['products'][['product_id','product_name','category']], on='product_id')

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Stock Value", f"₹{inv['stock_value'].sum()/1e7:.2f} Cr")
c2.metric("Total Units", f"{inv['closing_stock'].sum()/1e6:.2f}M")
c3.metric("SKUs Tracked", inv['product_id'].nunique())
c4.metric("Warehouses", inv['warehouse'].nunique())

# Stock by warehouse
st.subheader("🏭 Stock by Warehouse")
wh = inv.groupby('warehouse')['stock_value'].sum().reset_index()
fig = px.bar(wh, x='warehouse', y='stock_value', color='warehouse', title='Stock Value by Warehouse')
st.plotly_chart(fig, use_container_width=True)

# Stock movement trend
st.subheader("📊 Daily Stock Movement")
movement = inv.groupby('snapshot_date')[['received', 'dispatched']].sum().reset_index()
fig = px.line(movement, x='snapshot_date', y=['received', 'dispatched'], title='Daily Stock Movement')
st.plotly_chart(fig, use_container_width=True)

# Slow movers
st.subheader("🐢 Slow Moving SKUs")
# Calculate turnover as dispatched/received
turnover = inv.groupby(['product_id', 'product_name']).agg(
    received=('received', 'sum'),
    dispatched=('dispatched', 'sum'),
    avg_stock=('closing_stock', 'mean')
).reset_index()
turnover['turnover_ratio'] = turnover['dispatched'] / turnover['avg_stock']
slow = turnover.sort_values('turnover_ratio').head(20)
st.dataframe(slow, use_container_width=True)
