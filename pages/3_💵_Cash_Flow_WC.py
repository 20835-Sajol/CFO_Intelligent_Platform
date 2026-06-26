import streamlit as st
import plotly.express as px
import plotly.graph_objects as go  # <-- ADDED THIS IMPORT
import pandas as pd
from utils.data_loader import load_data
from utils.kpi_calculator import working_capital_metrics
from plotly.subplots import make_subplots


st.set_page_config(page_title="Cash Flow & Working Capital", page_icon="💵", layout="wide")
st.title("💵 Cash Flow & Working Capital")

data = load_data()
ar = data['receivables']

# KPIs
metrics = working_capital_metrics(ar)
c1, c2, c3, c4 = st.columns(4)
for i, (k, v) in enumerate(metrics.items()):
    [c1, c2, c3, c4][i].metric(k, v)

st.markdown("---")

# Aging analysis
st.subheader("📅 Receivables Aging")
def age_bucket(days):
    if days <= 0: return 'Current'
    elif days <= 30: return '1-30 days'
    elif days <= 60: return '31-60 days'
    elif days <= 90: return '61-90 days'
    else: return '90+ days'

ar['aging_bucket'] = ar['aging_days'].apply(age_bucket)
aging = ar[ar['status'] != 'Paid'].groupby('aging_bucket')['outstanding'].sum().reset_index()

fig = px.bar(aging, x='aging_bucket', y='outstanding', 
             color='aging_bucket', title='Aging Analysis (Outstanding Amount)',
             color_discrete_sequence=px.colors.diverging.RdYlGn_r)
st.plotly_chart(fig, use_container_width=True)

# Collection efficiency
st.subheader("📈 Collection Trend")
ar['month'] = ar['invoice_date'].dt.to_period('M').astype(str)
collection = ar.groupby('month').agg(
    invoiced=('invoice_amount', 'sum'),
    collected=('paid_amount', 'sum')
).reset_index()
collection['collection_rate'] = (collection['collected'] / collection['invoiced']) * 100

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=collection['month'], y=collection['invoiced'], name='Invoiced'))
fig.add_trace(go.Bar(x=collection['month'], y=collection['collected'], name='Collected'))
fig.add_trace(go.Scatter(x=collection['month'], y=collection['collection_rate'], 
                         name='Collection %', mode='lines+markers', yaxis='y2'))
fig.update_layout(title='Monthly Invoiced vs Collected', barmode='group', template='plotly_white')
fig.update_yaxes(title="Amount", secondary_y=False)
fig.update_yaxes(title="Collection %", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# Top overdue
st.subheader("🚨 Top Overdue Outlets")
overdue = ar[ar['status'] != 'Paid'].groupby('outlet_id').agg(
    outstanding=('outstanding', 'sum'),
    avg_aging=('aging_days', 'mean')
).sort_values('outstanding', ascending=False).head(20).reset_index()
overdue = overdue.merge(data['outlets'][['outlet_id', 'outlet_name']], on='outlet_id')
st.dataframe(overdue, use_container_width=True)
