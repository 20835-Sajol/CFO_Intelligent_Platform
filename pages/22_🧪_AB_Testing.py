import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from utils.data_loader import load_data
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="A/B Testing", page_icon="🧪", layout="wide")
st.title("🧪 Bayesian A/B Testing Framework")
st.markdown("### Promo Experiments with Probabilistic Decisions")

data = load_data()
sales = data['sales'].copy()

# ============================================
# A/B TEST SETUP
# ============================================
st.subheader("🎯 Experiment Configuration")

test_type = st.selectbox("Test Type", [
    "Promotion A vs B",
    "Price Point A vs B",
    "Channel A vs B",
    "Campaign Creative A vs B"
])

c1, c2 = st.columns(2)
with c1:
    metric = st.selectbox("Primary Metric", 
                           ["Conversion Rate", "Average Order Value", "Revenue per Customer", "Quantity Sold"])
with c2:
    test_period = st.slider("Test Duration (days)", 7, 90, 30)

# ============================================
# GENERATE OR USE REAL A/B DATA
# ============================================
st.markdown("---")
st.subheader("📊 Experimental Data")

# Simulate A/B test based on sales data
np.random.seed(42)

# Sample outlets and randomly assign
outlets_sample = data['outlets'].sample(n=min(200, len(data['outlets'])))
outlets_sample = outlets_sample.reset_index(drop=True)
outlets_sample['variant'] = np.random.choice(['A (Control)', 'B (Treatment)'], size=len(outlets_sample))

# Calculate metrics per outlet
outlet_metrics = sales.groupby('outlet_id').agg(
    orders=('order_id', 'nunique'),
    revenue=('net_amount', 'sum'),
    quantity=('quantity', 'sum')
).reset_index()
outlet_metrics['aov'] = outlet_metrics['revenue'] / outlet_metrics['orders']

ab_data = outlets_sample.merge(outlet_metrics, on='outlet_id', how='left')

# Simulate treatment effect (10-20% lift)
treatment_mask = ab_data['variant'] == 'B (Treatment)'
ab_data.loc[treatment_mask, 'revenue'] *= np.random.uniform(1.10, 1.25, treatment_mask.sum())
ab_data.loc[treatment_mask, 'orders'] = (ab_data.loc[treatment_mask, 'orders'] * 
                                          np.random.uniform(1.05, 1.20, treatment_mask.sum())).astype(int)
ab_data.loc[treatment_mask, 'aov'] = ab_data.loc[treatment_mask, 'revenue'] / ab_data.loc[treatment_mask, 'orders']

# ============================================
# CLASSICAL FREQUENTIST TEST
# ============================================
st.subheader("📈 Frequentist Analysis")

variant_a = ab_data[ab_data['variant']=='A (Control)']
variant_b = ab_data[ab_data['variant']=='B (Treatment)']

if metric == "Conversion Rate":
    # Simulate conversion (order rate)
    a_conv = variant_a['orders'] > 0
    b_conv = variant_b['orders'] > 0
    
    # Ensure valid counts
    a_total = max(len(a_conv), 10)
    b_total = max(len(b_conv), 10)
    a_conv_count = max(a_conv.sum(), 5)
    b_conv_count = max(b_conv.sum(), 5)
    
    a_rate = a_conv_count / a_total
    b_rate = b_conv_count / b_total
    
    metric_a = a_rate
    metric_b = b_rate
    
elif metric == "Average Order Value":
    metric_a = variant_a['aov'].mean()
    metric_b = variant_b['aov'].mean()
    
elif metric == "Revenue per Customer":
    metric_a = variant_a['revenue'].mean()
    metric_b = variant_b['revenue'].mean()
    
else:  # Quantity Sold
    metric_a = variant_a['quantity'].mean()
    metric_b = variant_b['quantity'].mean()

# T-test
if metric == "Conversion Rate":
    from statsmodels.stats.proportion import proportions_ztest
    z_stat, p_value = proportions_ztest([a_conv_count, b_conv_count], [a_total, b_total])
    lift_pct = (b_rate - a_rate) / a_rate * 100
else:
    t_stat, p_value = stats.ttest_ind(variant_a['aov'].dropna() if metric == "Average Order Value" else
                                       variant_a['revenue'].dropna(),
                                       variant_b['aov'].dropna() if metric == "Average Order Value" else
                                       variant_b['revenue'].dropna())
    lift_pct = (metric_b - metric_a) / metric_a * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Variant A (Control)", f"{metric_a:.4f}" if metric == "Conversion Rate" else f"₹{metric_a:.2f}")
c2.metric("Variant B (Treatment)", f"{metric_b:.4f}" if metric == "Conversion Rate" else f"₹{metric_b:.2f}")
c3.metric("Lift", f"{lift_pct:+.2f}%")
c4.metric("P-value", f"{p_value:.4f}", "Significant" if p_value < 0.05 else "Not Significant")

# ============================================
# BAYESIAN A/B TEST
# ============================================
st.markdown("---")
st.subheader("🎲 Bayesian Analysis")

n_simulations = 100000

if metric == "Conversion Rate":
    # Beta distribution
    alpha_a, beta_a = a_conv_count + 1, a_total - a_conv_count + 1
    alpha_b, beta_b = b_conv_count + 1, b_total - b_conv_count + 1
    
    samples_a = np.random.beta(alpha_a, beta_a, n_simulations)
    samples_b = np.random.beta(alpha_b, beta_b, n_simulations)
    
else:
    # Normal approximation
    samples_a = np.random.normal(metric_a, metric_a * 0.1, n_simulations)
    samples_b = np.random.normal(metric_b, metric_b * 0.1, n_simulations)

# Probability B > A
prob_b_better = (samples_b > samples_a).mean()
# Expected lift
expected_lift = ((samples_b - samples_a) / (samples_a + 1e-9) * 100).mean()
# Credible interval for lift
lift_samples = (samples_b - samples_a) / (samples_a + 1e-9) * 100
lift_lower = np.percentile(lift_samples, 2.5)
lift_upper = np.percentile(lift_samples, 97.5)

c1, c2, c3 = st.columns(3)
c1.metric("P(B > A)", f"{prob_b_better*100:.1f}%")
c2.metric("Expected Lift", f"{expected_lift:+.2f}%")
c3.metric("95% Credible Interval", f"[{lift_lower:+.1f}%, {lift_upper:+.1f}%]")

# ============================================
# VISUALIZATIONS
# ============================================
c1, c2 = st.columns(2)

with c1:
    # Posterior distributions
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=samples_a, name='Variant A', opacity=0.7, 
                                marker_color='blue', nbinsx=50))
    fig.add_trace(go.Histogram(x=samples_b, name='Variant B', opacity=0.7,
                                marker_color='red', nbinsx=50))
    fig.update_layout(barmode='overlay', title=f'Posterior Distributions: {metric}',
                      xaxis_title='Metric Value', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Lift distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=lift_samples, nbinsx=50, marker_color='purple'))
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Effect")
    fig.add_vline(x=expected_lift, line_dash="dash", line_color="green", 
                  annotation_text=f"Expected: {expected_lift:.1f}%")
    fig.update_layout(title='Distribution of Lift %', xaxis_title='Lift %',
                      template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# DECISION FRAMEWORK
# ============================================
st.markdown("---")
st.subheader("🎯 Decision Recommendation")

threshold = st.slider("Minimum Acceptable Probability of B > A", 0.50, 0.99, 0.95)

col1, col2 = st.columns([1, 2])

with col1:
    if prob_b_better >= threshold:
        st.success(f"""
        ✅ **SHIP VARIANT B**
        
        Confidence: {prob_b_better*100:.1f}%
        Expected Lift: {expected_lift:+.2f}%
        """)
    else:
        st.warning(f"""
        ⚠️ **INSUFFICIENT EVIDENCE**
        
        Confidence: {prob_b_better*100:.1f}%
        Need: {threshold*100:.0f}%
        
        Recommend: Continue test or stick with A
        """)

with col2:
    # Risk analysis
    prob_negative_lift = (lift_samples < 0).mean() * 100
    prob_major_loss = (lift_samples < -10).mean() * 100
    
    st.write("**📊 Risk Analysis:**")
    st.write(f"- Probability of negative lift: **{prob_negative_lift:.1f}%**")
    st.write(f"- Probability of >10% loss: **{prob_major_loss:.1f}%**")
    
    if prob_negative_lift < 5:
        st.success("✅ Low risk of negative impact")
    elif prob_negative_lift < 20:
        st.warning("⚠️ Moderate risk - consider segment analysis")
    else:
        st.error("🚨 High risk - reconsider launch")

st.markdown("---")
st.info("""
**Bayesian A/B Testing Advantages:**
- 📊 Direct probability statements ("B is 87% likely better")
- 🎯 No p-value confusion
- 📈 Can stop test anytime without inflating false positive rate
- 🔄 Incorporates prior knowledge
- 💡 Better for business decision-making
""")
