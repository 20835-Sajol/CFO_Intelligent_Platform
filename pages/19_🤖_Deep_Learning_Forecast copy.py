import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import torch with error handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    torch_error = str(e)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
from utils.data_loader import load_data

st.set_page_config(page_title="Deep Learning Forecast", page_icon="🤖", layout="wide")
st.title("🤖 Deep Learning Forecasting")
st.markdown("### LSTM Neural Networks for Time Series")

# Check PyTorch
if not TORCH_AVAILABLE:
    st.error(f"""
    ❌ PyTorch could not be loaded. This is typically a Windows DLL issue.
    
    **Solution:** Reinstall PyTorch with CPU support:
    ```
    pip uninstall torch -y
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```
    
    Then restart the app.
    """)
    st.stop()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = load_data()
sales = data['sales'].copy()

# ============================================
# CONFIGURATION
# ============================================
st.sidebar.header("⚙️ DL Configuration")
target = st.sidebar.selectbox("Forecast", ["Overall Revenue", "By Category", "By Channel"])

if target == "By Category":
    cat = st.sidebar.selectbox("Category", sales['category'].unique())
    ts = sales[sales['category']==cat].groupby(sales['order_date'].dt.to_period('M'))['net_amount'].sum()
elif target == "By Channel":
    ch = st.sidebar.selectbox("Channel", sales['channel'].unique())
    ts = sales[sales['channel']==ch].groupby(sales['order_date'].dt.to_period('M'))['net_amount'].sum()
else:
    ts = sales.groupby(sales['order_date'].dt.to_period('M'))['net_amount'].sum()

ts.index = ts.index.to_timestamp()

lookback = st.sidebar.slider("Lookback Window (months)", 3, 24, 12)
forecast_horizon = st.sidebar.slider("Forecast Horizon", 1, 12, 6)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)

# ============================================
# LSTM MODEL
# ============================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ============================================
# PREPARE SEQUENCES
# ============================================
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# Normalize
scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()

X, y = create_sequences(ts_scaled, lookback)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# ============================================
# TRAIN MODEL
# ============================================
st.subheader("🧠 LSTM Training")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🚀 Train LSTM Model"):
        with st.spinner("Training..."):
            model = LSTMModel(input_size=1, hidden_size=50, num_layers=2).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            losses = []
            progress_bar = st.progress(0)
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs.squeeze(), y_train_t)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                progress_bar.progress((epoch+1)/epochs)
            
            # Save predictions
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_t).cpu().numpy().flatten()
                test_pred = model(X_test_t).cpu().numpy().flatten()
                future_input = torch.FloatTensor(ts_scaled[-lookback:].reshape(1, lookback, 1)).to(device)
                future_pred = []
                for _ in range(forecast_horizon):
                    pred = model(future_input).cpu().numpy()[0, 0]
                    future_pred.append(pred)
                    future_input = torch.cat([future_input[:, 1:, :], 
                                              torch.FloatTensor([[[pred]]]).to(device)], dim=1)
            
            st.session_state['lstm_results'] = {
                'train_pred': scaler.inverse_transform(train_pred.reshape(-1,1)).flatten(),
                'test_pred': scaler.inverse_transform(test_pred.reshape(-1,1)).flatten(),
                'future_pred': scaler.inverse_transform(np.array(future_pred).reshape(-1,1)).flatten(),
                'losses': losses,
                'mape': mean_absolute_percentage_error(y_test, test_pred) * 100
            }
            st.success("✅ Training complete!")

with col2:
    if 'lstm_results' in st.session_state:
        results = st.session_state['lstm_results']
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE", f"{results['mape']:.2f}%")
        c2.metric("Epochs", len(results['losses']))
        c3.metric("Final Loss", f"{results['losses'][-1]:.6f}")
        
        # Loss curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=results['losses'], mode='lines', name='Training Loss'))
        fig.update_layout(title='LSTM Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# VISUALIZATION
# ============================================
if 'lstm_results' in st.session_state:
    results = st.session_state['lstm_results']
    
    st.subheader("📈 Forecast Visualization")
    
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines+markers',
                              name='Actual', line=dict(color='black', width=3)))
    
    # Train predictions
    train_dates = ts.index[lookback:lookback+len(results['train_pred'])]
    fig.add_trace(go.Scatter(x=train_dates, y=results['train_pred'], mode='lines',
                              name='Train Pred', line=dict(color='blue', dash='dash')))
    
    # Test predictions
    test_dates = ts.index[lookback+len(results['train_pred']):]
    fig.add_trace(go.Scatter(x=test_dates, y=results['test_pred'], mode='lines+markers',
                              name='Test Pred', line=dict(color='orange', dash='dash')))
    
    # Future predictions
    future_idx = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), 
                                periods=forecast_horizon, freq='MS')
    fig.add_trace(go.Scatter(x=future_idx, y=results['future_pred'], mode='lines+markers',
                              name='Future Forecast', line=dict(color='red', dash='dot', width=3)))
    
    fig.update_layout(title='LSTM Time Series Forecast', template='plotly_white', height=500)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("""
**LSTM Advantages for FMCG:**
- 📚 Learns long-term dependencies
- 🎯 Handles non-linear patterns
- 🔄 Captures seasonality + trend + irregularity
- 📊 Better for complex multivariate forecasts
""")
