import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Survival Analysis", page_icon="🧬", layout="wide")
st.title("🧬 Customer Survival & Churn Analysis")

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
except ImportError:
    st.error("Install lifelines: `pip install lifelines`")
    st.stop()

data = load_data()
sales = data['sales'].copy()

# ============================================
# BUILD SURVIVAL DATASET
# ============================================
st.subheader("📊 Build Customer Survival Dataset")

# Last purchase per outlet
outlet_last_purchase = sales.groupby('outlet_id')['order_date'].max().reset_index()
outlet_last_purchase.columns = ['outlet_id', 'last_purchase']
outlet_last_purchase['days_since_last'] = (sales['order_date'].max() - outlet_last_purchase['last_purchase']).dt.days

# First purchase
outlet_first_purchase = sales.groupby('outlet_id')['order_date'].min().reset_index()
outlet_first_purchase.columns = ['outlet_id', 'first_purchase']

# Merge
survival_df = outlet_last_purchase.merge(outlet_first_purchase, on='outlet_id')
survival_df['tenure_days'] = (survival_df['last_purchase'] - survival_df['first_purchase']).dt.days

# Churned if no purchase in last 90 days
survival_df['churned'] = (survival_df['days_since_last'] > 90).astype(int)

# Add features
outlet_features = sales.groupby('outlet_id').agg(
    total_orders=('order_id', 'nunique'),
    total_revenue=('net_amount', 'sum'),
    avg_order_value=('net_amount', 'mean'),
    unique_products=('product_id', 'nunique')
).reset_index()

survival_df = survival_df.merge(outlet_features, on='outlet_id')
outlet_meta = data['outlets'][['outlet_id','outlet_type','region','credit_days']].rename(
    columns={'outlet_type': 'channel'}
)
survival_df = survival_df.merge(outlet_meta, on='outlet_id')

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", len(survival_df))
c2.metric("Churned (90+ days)", f"{survival_df['churned'].sum()}")
c3.metric("Churn Rate", f"{survival_df['churned'].mean()*100:.1f}%")
c4.metric("Avg Tenure", f"{survival_df['tenure_days'].mean():.0f} days")

# ============================================
# KAPLAN-MEIER SURVIVAL CURVES
# ============================================
st.markdown("---")
st.subheader("📈 Kaplan-Meier Survival Analysis")

kmf = KaplanMeierFitter()
kmf.fit(survival_df['tenure_days'], event_observed=survival_df['churned'])

fig = go.Figure()
fig.add_trace(go.Scatter(x=kmf.survival_function_.index, 
                          y=kmf.survival_function_['KM_estimate'],
                          mode='lines', name='Overall Survival',
                          line=dict(color='blue', width=3)))

# By channel
for channel in survival_df['channel'].unique():
    mask = survival_df['channel'] == channel
    kmf_ch = KaplanMeierFitter()
    kmf_ch.fit(survival_df[mask]['tenure_days'], event_observed=survival_df[mask]['churned'])
    
    fig.add_trace(go.Scatter(x=kmf_ch.survival_function_.index,
                              y=kmf_ch.survival_function_['KM_estimate'],
                              mode='lines', name=channel))

fig.update_layout(title='Customer Survival Curve by Channel',
                  xaxis_title='Days Since First Purchase',
                  yaxis_title='Survival Probability',
                  template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

# Median survival time
median_survival = kmf.median_survival_time_
st.metric("📊 Median Customer Lifetime", f"{median_survival:.0f} days")

# ============================================
# COX PROPORTIONAL HAZARDS MODEL
# ============================================
st.markdown("---")
st.subheader("⚠️ Cox PH Model - Churn Risk Factors")

# Prepare data for Cox
cox_df = survival_df[['tenure_days', 'churned', 'total_orders', 'total_revenue', 
                       'avg_order_value', 'unique_products', 'credit_days']].copy()

# Normalize
for col in ['total_revenue', 'avg_order_value']:
    cox_df[col] = np.log1p(cox_df[col])

# Fit model
cph = CoxPHFitter()
cph.fit(cox_df, duration_col='tenure_days', event_col='churned')

# Show coefficients
cox_summary = cph.summary.reset_index()
cox_summary = cox_summary.rename(columns={cox_summary.columns[0]: 'covariate'})
cox_summary['Hazard Ratio'] = np.exp(cox_summary['coef'])
cox_summary['Risk Level'] = cox_summary['Hazard Ratio'].apply(
    lambda x: '🔴 High Risk' if x > 1.2 else ('🟢 Protective' if x < 0.8 else '🟡 Neutral')
)

c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(cox_summary[['covariate','coef','Hazard Ratio','p','Risk Level']].style.format({
        'coef': '{:.3f}',
        'Hazard Ratio': '{:.3f}',
        'p': '{:.4f}'
    }), use_container_width=True)

with c2:
    fig = px.bar(cox_summary, x='Hazard Ratio', y='covariate', orientation='h',
                 color='Hazard Ratio', color_continuous_scale='RdYlGn_r',
                 title='Hazard Ratios')
    fig.add_vline(x=1, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# CUSTOMER RISK SCORING
# ============================================
st.markdown("---")
st.subheader("🎯 Customer Risk Scoring")

# Calculate risk scores
risk_scores = cph.predict_partial_hazard(survival_df[['total_orders','total_revenue',
                                                       'avg_order_value','unique_products','credit_days']])
survival_df['risk_score'] = risk_scores.values

# Categorize - handle case where all values might be the same
try:
    survival_df['risk_category'] = pd.qcut(survival_df['risk_score'], q=3, 
                                            labels=['🟢 Low Risk', '🟡 Medium Risk', '🔴 High Risk'],
                                            duplicates='drop')
except ValueError:
    # If all scores are the same, assign all to medium risk
    survival_df['risk_category'] = '🟡 Medium Risk'

c1, c2, c3 = st.columns(3)
c1.metric("🟢 Low Risk", (survival_df['risk_category']=='🟢 Low Risk').sum())
c2.metric("🟡 Medium Risk", (survival_df['risk_category']=='🟡 Medium Risk').sum())
c3.metric("🔴 High Risk", (survival_df['risk_category']=='🔴 High Risk').sum())

# Show high risk customers
st.write("**🚨 Top 20 High-Risk Customers (Retention Priority):**")
high_risk = survival_df[survival_df['risk_category']=='🔴 High Risk'].nlargest(20, 'risk_score')
high_risk = high_risk.merge(data['outlets'][['outlet_id','outlet_name']], on='outlet_id')

st.dataframe(high_risk[['outlet_name','total_orders','total_revenue',
                          'tenure_days','days_since_last','risk_score']].style.format({
    'total_revenue': '₹{:,.0f}',
    'risk_score': '{:.3f}'
}), use_container_width=True)

# ============================================
# RETENTION RECOMMENDATIONS
# ============================================
st.markdown("---")
st.subheader("💡 Retention Strategies")

high_risk_count = (survival_df['risk_category']=='🔴 High Risk').sum()
potential_revenue_loss = survival_df[survival_df['risk_category']=='🔴 High Risk']['total_revenue'].sum()

st.error(f"""
🚨 **High Risk Customers Identified**
- {high_risk_count} customers at high churn risk
- Potential revenue loss: **₹{potential_revenue_loss/1e5:.1f} L**
- **Action**: Immediate retention campaign needed
""")

st.success("""
✅ **Recommended Retention Strategies**
1. 📞 Personal outreach to high-risk customers
2. 💰 Special reactivation offers (15-20% discount)
3. 📧 Re-engagement email/SMS campaign
4. 🎁 Loyalty rewards for repeat purchases
5. 📊 Monitor credit terms (high credit = higher churn)
""")
