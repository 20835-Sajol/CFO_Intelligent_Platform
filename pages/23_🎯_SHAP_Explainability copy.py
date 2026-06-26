import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from utils.data_loader import load_data
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SHAP Explainability", page_icon="🎯", layout="wide")
st.title("🎯 Model Explainability with SHAP")
st.markdown("### Understand WHAT drives your revenue/predictions")

try:
    import shap
except ImportError:
    st.error("Install SHAP: `pip install shap`")
    st.stop()

data = load_data()
sales = data['sales'].copy()
sales = sales.merge(data['products'][['product_id','category','brand','unit_cost','mrp']], on='product_id', suffixes=('','_p'))
sales = sales.merge(data['outlets'][['outlet_id']], on='outlet_id', suffixes=('','_o'))

# ============================================
# FEATURE ENGINEERING
# ============================================
sales['month'] = sales['order_date'].dt.month
sales['quarter'] = sales['order_date'].dt.quarter
sales['day_of_week'] = sales['order_date'].dt.dayofweek
sales['is_weekend'] = sales['day_of_week'].isin([5,6]).astype(int)
sales['price_ratio'] = sales['unit_price'] / sales['mrp']
sales['discount_value'] = sales['discount_amount'] / sales['unit_price']

# Encode categoricals
sales_encoded = pd.get_dummies(sales, columns=['category','brand','channel','region'], drop_first=True)

feature_cols = [c for c in sales_encoded.columns if c not in 
                ['order_id','order_date','outlet_id','product_id','rep_id','scheme_id',
                 'net_amount','gross_amount','cogs','gross_profit','tax_amount']]

X = sales_encoded[feature_cols].fillna(0)
y = sales_encoded['net_amount']

# Sample for performance
sample_idx = np.random.choice(len(X), min(2000, len(X)), replace=False)
X_sample = X.iloc[sample_idx]
y_sample = y.iloc[sample_idx]

# ============================================
# TRAIN MODEL
# ============================================
st.subheader("🌲 Train Black-Box Model")

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

if st.button("🚀 Train Random Forest"):
    with st.spinner("Training model..."):
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        st.session_state['rf_model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['X_train'] = X_train
        st.session_state['model_score'] = score
        st.success(f"✅ Model R² Score: {score:.4f}")

# ============================================
# SHAP ANALYSIS
# ============================================
if 'rf_model' in st.session_state:
    model = st.session_state['rf_model']
    X_test = st.session_state['X_test']
    
    st.markdown("---")
    st.subheader("🔍 SHAP Feature Importance")
    
    with st.spinner("Computing SHAP values..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False, max_display=15)
        st.pyplot(fig)
        plt.clf()
    
    # Feature importance table
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean |SHAP|', ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Top 15 Features:**")
        st.dataframe(feature_importance.head(15).style.format({'Mean |SHAP|': '{:.4f}'}),
                     use_container_width=True)
    
    with col2:
        fig = px.bar(feature_importance.head(15), x='Mean |SHAP|', y='Feature',
                     orientation='h', color='Mean |SHAP|',
                     color_continuous_scale='Viridis',
                     title='Feature Importance (SHAP)')
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # INDIVIDUAL PREDICTION EXPLANATION
    # ============================================
    st.markdown("---")
    st.subheader("🎯 Explain Individual Predictions")
    
    sample_idx = st.slider("Select Sample to Explain", 0, len(X_test)-1, 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[sample_idx],
                                          base_values=explainer.expected_value,
                                          data=X_test.iloc[sample_idx]),
                         max_display=15, show=False)
    st.pyplot(fig)
    plt.clf()
    
    # Force plot
    st.write(f"**Prediction:** ₹{model.predict(X_test.iloc[[sample_idx]])[0]:,.2f}")
    expected_val = explainer.expected_value if isinstance(explainer.expected_value, (int, float)) else explainer.expected_value.item()
    st.write(f"**Base Value:** ₹{expected_val:,.2f}")
    st.write(f"**Difference:** ₹{model.predict(X_test.iloc[[sample_idx]])[0] - expected_val:+,.2f}")

# ============================================
# DEPENDENCE PLOTS
# ============================================
if 'rf_model' in st.session_state:
    st.markdown("---")
    st.subheader("📈 Feature Dependence Analysis")
    
    feature_to_plot = st.selectbox("Select Feature", X_test.columns[:20])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(feature_to_plot, shap_values, X_test, show=False)
    st.pyplot(fig)
    plt.clf()
    
    st.info("""
    **SHAP Value Interpretation:**
    - **Positive SHAP**: Feature increases prediction
    - **Negative SHAP**: Feature decreases prediction
    - **Magnitude**: How much that feature contributes
    - **Base value**: Average model output
    - **Sum of SHAP + Base = Prediction**
    """)

# ============================================
# BUSINESS INSIGHTS
# ============================================
st.markdown("---")
st.subheader("💡 Business Insights from SHAP")

if 'rf_model' in st.session_state:
    top_5_features = feature_importance.head(5)['Feature'].tolist()
    
    insights = {
        'has_promotion': "🎯 Promotions are driving revenue — scale successful campaigns",
        'discount_pct': "💰 Discount levels significantly impact sales — optimize discount strategy",
        'price_ratio': "📊 Price positioning matters — monitor competitive pricing",
        'quantity': "📦 Volume is a key driver — focus on high-volume outlets",
        'category': "📦 Category mix matters — optimize portfolio",
        'channel': "🏪 Channel selection critical — double down on best channels",
        'region': "🌍 Regional differences exist — customize strategies",
        'month': "📅 Seasonality is strong — plan inventory and promos accordingly"
    }
    
    for feat in top_5_features:
        for key, insight in insights.items():
            if key in feat.lower():
                st.success(insight)
                break
