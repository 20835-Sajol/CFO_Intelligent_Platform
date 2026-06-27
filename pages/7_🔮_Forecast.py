import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import load_data
from prophet import Prophet

st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Revenue Forecast")

# Load and process data
data = load_data()
sales = data['sales'].copy()

# Aggregate monthly
monthly = sales.groupby(sales['order_date'].dt.to_period('M'))['net_amount'].sum().reset_index()
monthly.columns = ['ds', 'y']
monthly['ds'] = monthly['ds'].dt.to_timestamp()

st.subheader("📈 Historical vs Forecast")
periods = st.slider("Forecast Months", 3, 24, 12)

# Pristine Initialization without any environment or logger hacks
m = Prophet(
    growth='linear',
    yearly_seasonality=True, 
    weekly_seasonality=False, # Disabled: Monthly points do not have weekly patterns
    daily_seasonality=False
)

# Fit the model
m.fit(monthly)

# Generate future dates (Using MS for Month Start to match historical dates)
future = m.make_future_dataframe(periods=periods, freq='MS')
forecast = m.predict(future)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly['ds'], y=monthly['y'], mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                         mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                         mode='lines', line=dict(width=0), fill='tonexty',
                         fillcolor='rgba(0,100,80,0.2)', name='Confidence Interval'))
fig.update_layout(title=f'Revenue Forecast - Next {periods} Months', template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

# KPI
future_only = forecast.tail(periods)
total_forecast = future_only['yhat'].sum()
st.metric(f"📊 Forecasted Revenue (Next {periods} Months)", f"₹{total_forecast/1e7:.2f} Cr")

# Components
st.subheader("🔍 Forecast Components")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)