import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.data_loader import load_data
from utils.kpi_calculator import calculate_kpis

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CFO Intelligent Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### CFO Intelligent Platform v2.0\nFMCG Analytics | Powered by AI"
    }
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px;
    }
    
    /* Subheader */
    .subheader {
        color: #666;
        text-align: center;
        font-size: 16px;
        margin-bottom: 20px;
    }
    
    /* Metric Cards */
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Cards */
    .module-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 10px 0;
        transition: transform 0.2s;
    }
    
    .module-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .module-card h4 {
        color: #667eea;
        margin: 0 0 8px 0;
    }
    
    .module-card p {
        color: #666;
        font-size: 13px;
        margin: 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA WITH SESSION STATE
# ============================================
@st.cache_data(ttl=3600, show_spinner="Loading datasets...")
def load_all_data():
    return load_data()

if 'data_loaded' not in st.session_state:
    with st.spinner('🔄 Loading comprehensive FMCG datasets...'):
        st.session_state['data'] = load_all_data()
        st.session_state['data_loaded'] = True

data = st.session_state['data']

# ============================================
# HEADER
# ============================================
st.markdown('<p class="main-header">💼 CFO Intelligent Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">FMCG Sales & Financial Analytics | AI-Powered Decision Intelligence</p>', 
            unsafe_allow_html=True)

# ============================================
# SIDEBAR - GLOBAL CONFIGURATION
# ============================================
with st.sidebar:
    st.markdown("### 🎛️ Global Configuration")
    
    # Date Range
    date_range = st.date_input(
        "📅 Date Range",
        value=[data['sales']['order_date'].min(), data['sales']['order_date'].max()],
        min_value=data['sales']['order_date'].min(),
        max_value=data['sales']['order_date'].max(),
        key='global_date_range'
    )
    
    # Dimension Filters
    st.markdown("#### 🌍 Geographic & Channel Filters")
    regions = st.multiselect(
        "Regions",
        options=sorted(data['outlets']['region'].unique()),
        default=sorted(data['outlets']['region'].unique()),
        key='global_regions'
    )
    
    channels = st.multiselect(
        "Channels",
        options=sorted(data['outlets']['outlet_type'].unique()),
        default=sorted(data['outlets']['outlet_type'].unique()),
        key='global_channels'
    )
    
    # Store in session state for cross-page access
    st.session_state['filters'] = {
        'date_range': date_range,
        'regions': regions,
        'channels': channels
    }
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Outlets", f"{len(data['outlets']):,}")
    st.metric("Total SKUs", f"{len(data['products']):,}")
    st.metric("Sales Reps", f"{len(data['sales_reps']):,}")
    
    st.markdown("---")
    st.markdown("### 🔔 Active Alerts")
    # Calculate active alerts
    overdue = data['receivables'][data['receivables']['status'] == 'Overdue']
    if len(overdue) > 0:
        st.error(f"🚨 {len(overdue)} overdue invoices")
    else:
        st.success("✅ All clear")
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ============================================
# APPLY GLOBAL FILTERS
# ============================================
def apply_global_filters(data):
    """Apply filters to sales data"""
    sales = data['sales'].copy()
    filters = st.session_state.get('filters', {})
    
    if len(filters.get('date_range', [])) == 2:
        sales = sales[
            (sales['order_date'].dt.date >= filters['date_range'][0]) &
            (sales['order_date'].dt.date <= filters['date_range'][1])
        ]
    
    if filters.get('regions'):
        valid_outlets = data['outlets'][data['outlets']['region'].isin(filters['regions'])]['outlet_id']
        sales = sales[sales['outlet_id'].isin(valid_outlets)]
    
    if filters.get('channels'):
        valid_outlets = data['outlets'][data['outlets']['outlet_type'].isin(filters['channels'])]['outlet_id']
        sales = sales[sales['outlet_id'].isin(valid_outlets)]
    
    return sales

sales_filtered = apply_global_filters(data)

# ============================================
# EXECUTIVE OVERVIEW
# ============================================
st.markdown("---")
st.markdown("## 📊 Executive Overview")

# Top KPI Row
if len(sales_filtered) > 0:
    kpis = calculate_kpis(sales_filtered)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("💰 Revenue", kpis['Total Revenue'])
    col2.metric("📈 Gross Profit", kpis['Gross Profit'])
    col3.metric("📊 Margin", kpis['Gross Margin %'])
    col4.metric("🛒 Orders", kpis['Total Orders'])
    col5.metric("📦 SKUs", kpis['Active SKUs'])
    col6.metric("🏪 Outlets", kpis['Active Outlets'])
    
    # Secondary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_qty = sales_filtered['quantity'].sum()
    aov = sales_filtered['net_amount'].sum() / sales_filtered['order_id'].nunique()
    total_discount = sales_filtered['discount_amount'].sum()
    promo_pct = (sales_filtered['has_promotion'].sum() / len(sales_filtered) * 100)
    
    col1.metric("📦 Units Sold", f"{total_qty/1e6:.2f}M")
    col2.metric("💵 Avg Order Value", f"₹{aov:,.0f}")
    col3.metric("🏷️ Total Discount Given", f"₹{total_discount/1e5:.2f}L")
    col4.metric("🎯 Promo Orders", f"{promo_pct:.1f}%")
    
else:
    st.warning("⚠️ No data matches current filters. Please adjust filters.")

# ============================================
# QUICK INSIGHTS
# ============================================
if len(sales_filtered) > 0:
    st.markdown("---")
    st.markdown("## 💡 Quick Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <h4>🏆 Top Channel</h4>
        </div>
        """, unsafe_allow_html=True)
        top_channel = sales_filtered.groupby('channel')['net_amount'].sum().nlargest(1)
        if len(top_channel) > 0:
            channel_name = top_channel.index[0]
            channel_rev = top_channel.values[0]
            st.metric(channel_name, f"₹{channel_rev/1e7:.2f} Cr")
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <h4>🌟 Top Region</h4>
        </div>
        """, unsafe_allow_html=True)
        top_region = sales_filtered.groupby('region')['net_amount'].sum().nlargest(1)
        if len(top_region) > 0:
            region_name = top_region.index[0]
            region_rev = top_region.values[0]
            st.metric(region_name, f"₹{region_rev/1e7:.2f} Cr")
    
    with col3:
        st.markdown("""
        <div class="module-card">
            <h4>📦 Top Category</h4>
        </div>
        """, unsafe_allow_html=True)
        sales_with_cat = sales_filtered.merge(
            data['products'][['product_id', 'category']], on='product_id'
        )
        top_cat = sales_with_cat.groupby('category')['net_amount'].sum().nlargest(1)
        if len(top_cat) > 0:
            cat_name = top_cat.index[0]
            cat_rev = top_cat.values[0]
            st.metric(cat_name, f"₹{cat_rev/1e7:.2f} Cr")

# ============================================
# PLATFORM MODULES (24 PAGES)
# ============================================
st.markdown("---")
st.markdown("## 🎯 Platform Modules")
st.markdown("*Navigate via sidebar → Pages*")

# Group 1: Core Analytics
st.markdown("### 📊 Core Analytics")
col1, col2, col3, col4 = st.columns(4)

modules_core = [
    ("📊 Revenue Analytics", "Sales trends, channel/region/SKU performance"),
    ("💰 Profitability", "Gross/Net margin, P&L by dimension"),
    ("💵 Cash Flow & WC", "DSO/DPO/DIO, collections, aging"),
    ("📦 Inventory", "Stock health, slow-movers, turnover")
]

for i, (title, desc) in enumerate(modules_core):
    with [col1, col2, col3, col4][i % 4]:
        st.markdown(f"""
        <div class="module-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Group 2: Business Performance
st.markdown("### 📈 Business Performance")
col1, col2, col3, col4 = st.columns(4)

modules_business = [
    ("🎯 Promotion ROI", "Campaign effectiveness, scheme performance"),
    ("📋 Budget vs Actual", "Variance analysis, achievement tracking"),
    ("📤 Export Reports", "PDF & Excel report generation"),
    ("🔔 Alerts", "Real-time monitoring & notifications")
]

for i, (title, desc) in enumerate(modules_business):
    with [col1, col2, col3, col4][i % 4]:
        st.markdown(f"""
        <div class="module-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Group 3: AI & Forecasting
st.markdown("### 🤖 AI & Machine Learning")
col1, col2, col3, col4 = st.columns(4)

modules_ai = [
    ("🔮 Forecast", "AI-based revenue & cash forecasting"),
    ("🧠 Advanced Forecasting", "Multi-model ensemble (Prophet/SARIMA/XGBoost)"),
    ("🤖 Deep Learning", "LSTM neural networks"),
    ("🎲 Bayesian Forecast", "Probabilistic predictions with PyMC")
]

for i, (title, desc) in enumerate(modules_ai):
    with [col1, col2, col3, col4][i % 4]:
        st.markdown(f"""
        <div class="module-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Group 4: Advanced Analytics
st.markdown("### 🔬 Advanced Analytics")
col1, col2, col3, col4 = st.columns(4)

modules_advanced = [
    ("🤖 AI/ML Module", "RFM segmentation, recommendations"),
    ("📊 Advanced Viz", "Sankey, Treemap, Sunburst, Cohort"),
    ("🚨 Anomaly & Risk", "Outliers, bad debts, fraud signals"),
    ("🎯 SHAP Explainability", "Model interpretability")
]

for i, (title, desc) in enumerate(modules_advanced):
    with [col1, col2, col3, col4][i % 4]:
        st.markdown(f"""
        <div class="module-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Group 5: Data Science
st.markdown("### 🧪 Data Science & Causal")
col1, col2, col3, col4 = st.columns(4)

modules_ds = [
    ("💎 CLV Prediction", "Customer lifetime value modeling"),
    ("📐 Price Elasticity", "Pricing optimization"),
    ("🧪 Causal Inference", "True promo impact (DiD)"),
    ("🧪 A/B Testing", "Bayesian experiment framework")
]

for i, (title, desc) in enumerate(modules_ds):
    with [col1, col2, col3, col4][i % 4]:
        st.markdown(f"""
        <div class="module-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Group 6: Specialized
st.markdown("### 🎯 Specialized Analytics")
col1, col2, col3, col4 = st.columns(4)

modules_special = [
    ("🕸️ Graph Analytics", "Distribution network analysis"),
    ("🧬 Survival Analysis", "Customer churn prediction"),
    ("💬 NLP Feedback", "Sentiment analysis from reviews"),
    ("🎰 Multi-Armed Bandit", "Dynamic pricing optimization")
]

for i, (title, desc) in enumerate(modules_special):
    with [col1, col2, col3, col4][i % 4]:
        st.markdown(f"""
        <div class="module-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>CFO Intelligent Platform v2.0</strong> | 24 Modules | Powered by AI/ML</p>
    <p>📊 Streamlit • 🐍 Python • 🤖 Scikit-learn • 🧠 PyTorch • 🎲 PyMC</p>
</div>
""", unsafe_allow_html=True)
