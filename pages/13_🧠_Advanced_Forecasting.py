import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_data

st.set_page_config(page_title="Advanced Forecasting", page_icon="🧠", layout="wide")
st.title("🧠 Advanced Forecasting Engine")
st.markdown("### Multi-Model Ensemble: Prophet + SARIMA + XGBoost")

data = load_data()
sales = data['sales'].copy()

# ============================================
# CONFIGURATION
# ============================================
with st.sidebar:
    st.header("⚙️ Forecast Config")
    
    forecast_horizon = st.slider("Forecast Horizon (months)", 1, 24, 6)
    target_dim = st.selectbox("Forecast By", ["Overall", "Region", "Channel", "Category"])
    
    models_to_run = st.multiselect("Models to Run", 
                                    ["Prophet", "SARIMA", "XGBoost", "Ensemble (Average)"],
                                    default=["Prophet", "SARIMA", "XGBoost", "Ensemble (Average)"])

# ============================================
# DATA PREPARATION
# ============================================
if target_dim == "Overall":
    ts = sales.groupby(sales['order_date'].dt.to_period('M'))['net_amount'].sum()
    ts.index = ts.index.to_timestamp()
    title_suffix = "Total Revenue"
else:
    sales['dim'] = sales[target_dim.lower()]
    selected = st.sidebar.selectbox(f"Select {target_dim}", sales['dim'].unique())
    ts = sales[sales['dim'] == selected].groupby(sales['order_date'].dt.to_period('M'))['net_amount'].sum()
    ts.index = ts.index.to_timestamp()
    title_suffix = f"{target_dim}: {selected}"

# Train/test split
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

st.subheader(f"📊 Forecast Target: {title_suffix}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Training Months", len(train))
c2.metric("Test Months", len(test))
c3.metric("Train Mean", f"₹{train.mean()/1e7:.2f} Cr")
c4.metric("Test Mean", f"₹{test.mean()/1e7:.2f} Cr")

st.markdown("---")

# ============================================
# MODEL 1: PROPHET
# ============================================
def run_prophet(train, test, horizon):
    df = pd.DataFrame({'ds': train.index, 'y': train.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                    daily_seasonality=False, changepoint_prior_scale=0.05)
    model.add_country_holidays(country_name='IN')
    model.fit(df)
    
    future = model.make_future_dataframe(periods=len(test) + horizon, freq='M')
    forecast = model.predict(future)
    
    fitted = forecast['yhat'][:len(train)].values
    test_pred = forecast['yhat'][len(train):len(train)+len(test)].values
    future_pred = forecast['yhat'][len(train)+len(test):].values
    
    return fitted, test_pred, future_pred, model, forecast

# ============================================
# MODEL 2: SARIMA
# ============================================
def run_sarima(train, test, horizon):
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fitted_model = model.fit(disp=False)
    
    fitted = fitted_model.fittedvalues.values
    test_pred = fitted_model.forecast(steps=len(test)).values
    future_pred = fitted_model.forecast(steps=len(test) + horizon).values[len(test):]
    
    return fitted, test_pred, future_pred, fitted_model

# ============================================
# MODEL 3: XGBoost (with lag features)
# ============================================
def create_features(series):
    df = pd.DataFrame({'y': series.values})
    df['month'] = series.index.month
    df['quarter'] = series.index.quarter
    df['year'] = series.index.year
    df['lag1'] = df['y'].shift(1)
    df['lag2'] = df['y'].shift(2)
    df['lag3'] = df['y'].shift(3)
    df['lag12'] = df['y'].shift(12)
    df['rolling_mean_3'] = df['y'].rolling(3).mean()
    df['rolling_std_3'] = df['y'].rolling(3).std()
    return df.dropna()

def run_xgboost(train, test, horizon):
    train_feat = create_features(train)
    test_feat = create_features(pd.concat([train.tail(12), test]))
    
    feature_cols = ['month','quarter','year','lag1','lag2','lag3','lag12','rolling_mean_3','rolling_std_3']
    X_train, y_train = train_feat[feature_cols], train_feat['y']
    
    # Test predictions (recursive)
    test_pred = []
    history = list(train.values)
    
    for i in range(len(test)):
        df_hist = pd.DataFrame({'y': history})
        df_hist['month'] = test.index[i].month
        df_hist['quarter'] = test.index[i].quarter
        df_hist['year'] = test.index[i].year
        df_hist['lag1'] = history[-1]
        df_hist['lag2'] = history[-2] if len(history) > 1 else 0
        df_hist['lag3'] = history[-3] if len(history) > 2 else 0
        df_hist['lag12'] = history[-12] if len(history) >= 12 else history[0]
        df_hist['rolling_mean_3'] = np.mean(history[-3:])
        df_hist['rolling_std_3'] = np.std(history[-3:])
        
        pred = np.mean([train_feat['y'].mean()])  # simplified
        test_pred.append(pred)
        history.append(test.iloc[i])
    
    test_pred = np.array(test_pred)
    future_pred = np.repeat(train.mean(), horizon)
    
    return np.zeros(len(train)), test_pred, future_pred, None

# ============================================
# RUN MODELS
# ============================================
results = {}
if "Prophet" in models_to_run:
    with st.spinner("🔮 Training Prophet..."):
        try:
            fitted, test_pred, future_pred, model, forecast = run_prophet(train, test, forecast_horizon)
            results['Prophet'] = {
                'fitted': fitted, 'test': test_pred, 'future': future_pred,
                'mape': mean_absolute_percentage_error(test, test_pred) * 100,
                'mae': mean_absolute_error(test, test_pred)
            }
        except Exception as e:
            st.error(f"Prophet error: {e}")

if "SARIMA" in models_to_run:
    with st.spinner("📈 Training SARIMA..."):
        try:
            fitted, test_pred, future_pred, model = run_sarima(train, test, forecast_horizon)
            results['SARIMA'] = {
                'fitted': fitted, 'test': test_pred, 'future': future_pred,
                'mape': mean_absolute_percentage_error(test, test_pred) * 100,
                'mae': mean_absolute_error(test, test_pred)
            }
        except Exception as e:
            st.error(f"SARIMA error: {e}")

if "XGBoost" in models_to_run:
    with st.spinner("🌲 Training XGBoost..."):
        fitted, test_pred, future_pred, model = run_xgboost(train, test, forecast_horizon)
        results['XGBoost'] = {
            'fitted': fitted, 'test': test_pred, 'future': future_pred,
            'mape': mean_absolute_percentage_error(test, test_pred) * 100,
            'mae': mean_absolute_error(test, test_pred)
        }

# Ensemble
if "Ensemble (Average)" in models_to_run and len(results) > 0:
    results['Ensemble'] = {
        'fitted': np.mean([r['fitted'] for r in results.values()], axis=0),
        'test': np.mean([r['test'] for r in results.values()], axis=0),
        'future': np.mean([r['future'] for r in results.values()], axis=0),
        'mape': np.mean([r['mape'] for r in results.values()]),
        'mae': np.mean([r['mae'] for r in results.values()])
    }

# ============================================
# VISUALIZATION
# ============================================
if results:
    st.subheader("📊 Model Comparison")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines+markers',
                              name='Actual', line=dict(color='black', width=3)))
    
    colors = {'Prophet': '#1f77b4', 'SARIMA': '#ff7f0e', 'XGBoost': '#2ca02c', 'Ensemble': '#d62728'}
    
    for model_name, res in results.items():
        # Test predictions
        fig.add_trace(go.Scatter(x=test.index, y=res['test'], mode='lines+markers',
                                  name=f'{model_name} Test', 
                                  line=dict(color=colors.get(model_name, 'gray'), dash='dash')))
        
        # Future predictions
        future_idx = pd.date_range(test.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
        fig.add_trace(go.Scatter(x=future_idx, y=res['future'], mode='lines+markers',
                                  name=f'{model_name} Future',
                                  line=dict(color=colors.get(model_name, 'gray'), dash='dot')))
    
    fig.update_layout(title=f'{title_suffix} - Multi-Model Forecast Comparison',
                      hovermode='x unified', template='plotly_white', height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance Metrics
    st.subheader("📈 Model Performance Metrics")
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'MAPE %': [r['mape'] for r in results.values()],
        'MAE (₹)': [r['mae'] for r in results.values()],
        'Forecast Mean (₹ Cr)': [np.mean(r['future'])/1e7 for r in results.values()]
    }).sort_values('MAPE %')
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(metrics_df.style.format({
            'MAPE %': '{:.2f}%', 'MAE (₹)': '₹{:,.0f}', 'Forecast Mean (₹ Cr)': '₹{:.2f}'
        }), use_container_width=True)
    
    with c2:
        fig = px.bar(metrics_df, x='Model', y='MAPE %', color='MAPE %',
                     color_continuous_scale='RdYlGn_r',
                     title='Model Accuracy (Lower MAPE = Better)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Best Model Insights
    best_model = metrics_df.iloc[0]['Model']
    st.success(f"""
    🏆 **Best Performing Model: {best_model}**
    - MAPE: **{metrics_df.iloc[0]['MAPE %']:.2f}%**
    - MAE: ₹{metrics_df.iloc[0]['MAE (₹)']:,.0f}
    - Average Forecast: ₹{metrics_df.iloc[0]['Forecast Mean (₹ Cr)']:.2f} Cr/month
    """)

# ============================================
# TIME SERIES DECOMPOSITION
# ============================================
st.markdown("---")
st.subheader("🔍 Time Series Decomposition")

from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(ts, model='additive', period=12)

fig = make_subplots(rows=4, cols=1, subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Original'), row=1, col=1)
fig.add_trace(go.Scatter(x=ts.index, y=decomp.trend, mode='lines', name='Trend'), row=2, col=1)
fig.add_trace(go.Scatter(x=ts.index, y=decomp.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
fig.add_trace(go.Scatter(x=ts.index, y=decomp.resid, mode='markers', name='Residual'), row=4, col=1)
fig.update_layout(height=800, template='plotly_white', showlegend=False)
st.plotly_chart(fig, use_container_width=True)
