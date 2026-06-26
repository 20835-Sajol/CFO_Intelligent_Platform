import streamlit as st
import pandas as pd
import numpy as np
import os
os.environ["PYTENSOR_FLAGS"] = "blas__ldflags="
import pymc as pm
import arviz as az
import plotly.graph_objects as go
from utils.data_loader import load_data
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Bayesian Forecast", page_icon="🎲", layout="wide")
st.title("🎲 Bayesian Forecasting with PyMC")
st.markdown("### Probabilistic Forecasts with Credible Intervals")

try:
    import pymc as pm
except ImportError:
    st.error("Install PyMC: `pip install pymc arviz`")
    st.stop()

data = load_data()
sales = data['sales'].copy()

# Configuration
target = st.selectbox("Forecast Target", ["Overall", "By Region", "By Category"])

if target == "By Region":
    region = st.selectbox("Region", sales['region'].unique())
    ts = sales[sales['region']==region].groupby(sales['order_date'].dt.to_period('M'))['net_amount'].sum()
elif target == "By Category":
    cat = st.selectbox("Category", sales['category'].unique())
    ts = sales[sales['category']==cat].groupby(sales['order_date'].dt.to_period('M'))['net_amount'].sum()
else:
    ts = sales.groupby(sales['order_date'].dt.to_period('M'))['net_amount'].sum()

ts.index = ts.index.to_timestamp()
ts = ts.reset_index()
ts.columns = ['date', 'value']

# Time index
ts['t'] = np.arange(len(ts))
ts['month'] = ts['date'].dt.month

# ============================================
# BAYESIAN STRUCTURAL TIME SERIES
# ============================================
st.subheader("🧮 Bayesian Structural Time Series")

if st.button("🎲 Run Bayesian Model"):
    with st.spinner("Sampling posterior (this may take a minute)..."):
        with pm.Model() as model:
            # Priors
            intercept = pm.Normal('intercept', mu=ts['value'].mean(), sigma=ts['value'].std())
            trend = pm.Normal('trend', mu=0, sigma=1000)
            
            # Seasonality (12 months)
            season_sigma = pm.HalfNormal('season_sigma', sigma=ts['value'].std()/2)
            season_raw = pm.Normal('season_raw', mu=0, sigma=1, shape=12)
            season = pm.Deterministic('season', season_raw * season_sigma)
            
            # Linear predictor
            t = ts['t'].values
            month = ts['month'].values
            
            mu = intercept + trend * t + season[month - 1]
            
            # Likelihood
            sigma = pm.HalfNormal('sigma', sigma=ts['value'].std())
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=ts['value'].values)
            
            # Sample
            trace = pm.sample(1000, tune=1000, cores=2, progressbar=False, return_inferencedata=True)
        
        st.session_state['bayes_trace'] = trace
        st.session_state['bayes_data'] = ts
        st.success("✅ Bayesian model fitted!")

# ============================================
# VISUALIZATION
# ============================================
if 'bayes_trace' in st.session_state:
    trace = st.session_state['bayes_trace']
    ts = st.session_state['bayes_data']
    
    # Posterior predictive
    with st.session_state['bayes_trace'].posterior:
        intercept = trace.posterior['intercept'].values.flatten()
        trend = trace.posterior['trend'].values.flatten()
        season = trace.posterior['season'].values.reshape(-1, 12)
        sigma = trace.posterior['sigma'].values.flatten()
    
    # Generate posterior predictive
    forecast_horizon = st.slider("Forecast Months", 1, 12, 6)
    future_t = np.arange(len(ts), len(ts) + forecast_horizon)
    future_month = [(ts['date'].iloc[-1].month + i) % 12 + 1 for i in range(1, forecast_horizon+1)]
    
    n_samples = 500
    predictions = []
    for i in range(n_samples):
        idx = np.random.randint(len(intercept))
        pred = intercept[idx] + trend[idx] * future_t + season[idx, np.array(future_month) - 1]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Mean and credible intervals
    pred_mean = predictions.mean(axis=0)
    pred_lower = np.percentile(predictions, 2.5, axis=0)
    pred_upper = np.percentile(predictions, 97.5, axis=0)
    pred_lower_50 = np.percentile(predictions, 25, axis=0)
    pred_upper_50 = np.percentile(predictions, 75, axis=0)
    
    # Plot
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(x=ts['date'], y=ts['value'], mode='lines+markers',
                              name='Actual', line=dict(color='black', width=3)))
    
    # Forecast
    future_dates = pd.date_range(ts['date'].iloc[-1] + pd.DateOffset(months=1), 
                                  periods=forecast_horizon, freq='MS')
    
    # 95% CI
    fig.add_trace(go.Scatter(x=future_dates, y=pred_upper, mode='lines',
                              line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=future_dates, y=pred_lower, mode='lines',
                              line=dict(width=0), fill='tonexty',
                              fillcolor='rgba(255,0,0,0.2)', name='95% CI'))
    
    # 50% CI
    fig.add_trace(go.Scatter(x=future_dates, y=pred_upper_50, mode='lines',
                              line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=future_dates, y=pred_lower_50, mode='lines',
                              line=dict(width=0), fill='tonexty',
                              fillcolor='rgba(255,0,0,0.4)', name='50% CI'))
    
    # Mean forecast
    fig.add_trace(go.Scatter(x=future_dates, y=pred_mean, mode='lines+markers',
                              name='Forecast (Mean)', line=dict(color='red', width=3)))
    
    fig.update_layout(title='Bayesian Forecast with Credible Intervals',
                      template='plotly_white', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Decision Support
    st.subheader("🎯 Probabilistic Decision Support")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Value (Mean)", f"₹{pred_mean.mean()/1e7:.2f} Cr")
    c2.metric("Worst Case (95% CI)", f"₹{pred_lower.mean()/1e7:.2f} Cr")
    c3.metric("Best Case (95% CI)", f"₹{pred_upper.mean()/1e7:.2f} Cr")
    
    # Probability of exceeding target
    target_value = st.number_input("Revenue Target (₹)", value=ts['value'].mean())
    prob_exceed = (predictions > target_value).mean() * 100
    
    st.metric(f"Probability of exceeding ₹{target_value/1e7:.2f} Cr", f"{prob_exceed:.1f}%")
    
    if prob_exceed > 70:
        st.success(f"✅ High confidence ({prob_exceed:.0f}%) of meeting target")
    elif prob_exceed > 40:
        st.warning(f"⚠️ Moderate confidence ({prob_exceed:.0f}%) - mitigation needed")
    else:
        st.error(f"🚨 Low confidence ({prob_exceed:.0f}%) - significant risk")

st.markdown("---")
st.info("""
**Bayesian Forecasting Advantages:**
- 📊 Full probability distribution (not just point estimates)
- 🎯 Quantifies uncertainty explicitly
- 💡 Enables risk-aware decisions
- 🔄 Incorporates prior knowledge
- 📈 Credible intervals (not confidence intervals)
""")
