import streamlit as st
import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data

st.set_page_config(page_title="CLV Prediction", page_icon="💎", layout="wide")
st.title("💎 Customer Lifetime Value Prediction")
st.markdown("### BG/NBD + Gamma-Gamma Model for CLV | Enhanced with Risk Analysis")

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
except ImportError:
    st.error("Install lifetimes: `pip install lifetimes`")
    st.stop()

data = load_data()
sales = data['sales'].copy()
sales = sales.merge(data['outlets'][['outlet_id','outlet_name']], on='outlet_id')

# ============================================
# RFM SUMMARY (PRESERVED - Original Code)
# ============================================
observation_period_end = sales['order_date'].max()
rfm = summary_data_from_transaction_data(
    sales, 'outlet_id', 'order_date', 'net_amount',
    observation_period_end=observation_period_end
)

st.subheader("📊 RFM Summary Statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", len(rfm))
c2.metric("Avg Frequency", f"{rfm['frequency'].mean():.2f}")
c3.metric("Avg Recency (days)", f"{rfm['recency'].mean():.0f}")
c4.metric("Avg Monetary (₹)", f"₹{rfm['monetary_value'].mean():,.0f}")

# ============================================
# BG/NBD MODEL (PRESERVED - Original Code)
# ============================================
st.subheader("🎯 BG/NBD Model - Future Purchases")

returning = rfm[rfm['frequency'] > 0]

with st.spinner("Training BG/NBD..."):
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(returning['frequency'], returning['recency'], returning['T'])

# Predict future purchases (next 90, 180, 365 days)
periods = {'90 days': 90, '180 days': 180, '365 days': 365}
predictions = pd.DataFrame({
    'Customer': returning.index,
    'Predicted_Purchases_90d': bgf.conditional_expected_number_of_purchases_up_to_time(90, returning['frequency'], returning['recency'], returning['T']),
    'Predicted_Purchases_180d': bgf.conditional_expected_number_of_purchases_up_to_time(180, returning['frequency'], returning['recency'], returning['T']),
    'Predicted_Purchases_365d': bgf.conditional_expected_number_of_purchases_up_to_time(365, returning['frequency'], returning['recency'], returning['T']),
})

predictions = predictions.merge(rfm, left_on='Customer', right_index=True)
predictions = predictions.merge(data['outlets'][['outlet_id','outlet_name']], 
                                  left_on='Customer', right_on='outlet_id')

st.dataframe(predictions[['outlet_name','frequency','monetary_value',
                           'Predicted_Purchases_90d','Predicted_Purchases_365d']].head(20), 
             use_container_width=True)

# ============================================
# GAMMA-GAMMA MODEL (PRESERVED - Original Code)
# ============================================
st.subheader("💰 Gamma-Gamma Model - Customer Value")

returning_filtered = returning[returning['monetary_value'] > 0]

with st.spinner("Training Gamma-Gamma..."):
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(returning_filtered['frequency'], returning_filtered['monetary_value'])

# Calculate CLV (12 months)
returning_filtered['predicted_avg_value'] = ggf.conditional_expected_average_profit(
    returning_filtered['frequency'], returning_filtered['monetary_value']
)

returning_filtered['CLV_12m'] = ggf.customer_lifetime_value(
    bgf, returning_filtered['frequency'], returning_filtered['recency'], 
    returning_filtered['T'], returning_filtered['monetary_value'],
    time=12, discount_rate=0.01
)

returning_filtered = returning_filtered.merge(data['outlets'][['outlet_id','outlet_name']], 
                                                left_index=True, right_on='outlet_id')

# ✅ Merge channel and region from sales data (Same as original)
outlet_channels = sales[['outlet_id', 'channel', 'region']].drop_duplicates()
returning_filtered = returning_filtered.merge(outlet_channels, on='outlet_id', how='left')

# CLV Segments
returning_filtered['CLV_Segment'] = pd.qcut(returning_filtered['CLV_12m'], 
                                              q=4, 
                                              labels=['🥉 Bronze', '🥈 Silver', '🥇 Gold', '💎 Platinum'])

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg CLV (12m)", f"₹{returning_filtered['CLV_12m'].mean():,.0f}")
c2.metric("Median CLV (12m)", f"₹{returning_filtered['CLV_12m'].median():,.0f}")
c3.metric("Top 10% CLV", f"₹{returning_filtered['CLV_12m'].quantile(0.9):,.0f}")
c4.metric("Total CLV (12m)", f"₹{returning_filtered['CLV_12m'].sum()/1e7:.2f} Cr")

# ============================================
# CLV VISUALIZATIONS (PRESERVED - Original Code)
# ============================================
st.subheader("📈 CLV Distribution")

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(returning_filtered, x='CLV_12m', nbins=50, 
                       color='CLV_Segment', title='CLV Distribution by Segment')
    st.plotly_chart(fig, use_container_width=True)

with c2:
    seg_summary = returning_filtered.groupby('CLV_Segment', observed=True).agg(
        count=('outlet_id','count'),
        avg_clv=('CLV_12m','mean'),
        total_clv=('CLV_12m','sum')
    ).reset_index()
    
    fig = px.bar(seg_summary, x='CLV_Segment', y='total_clv', color='CLV_Segment',
                 title='Total CLV Contribution by Segment')
    st.plotly_chart(fig, use_container_width=True)

# Top customers
st.subheader("🏆 Top 20 Customers by CLV")
top_clv = returning_filtered.nlargest(20, 'CLV_12m')[['outlet_name','channel','region','frequency','monetary_value','CLV_12m']]
st.dataframe(top_clv.style.format({'monetary_value':'₹{:,.0f}', 'CLV_12m':'₹{:,.0f}'}), use_container_width=True)

# Channel-wise CLV
st.subheader("🏪 Channel-wise Average CLV")
channel_clv = returning_filtered.groupby('channel')['CLV_12m'].agg(['mean','sum','count']).reset_index()
channel_clv.columns = ['Channel', 'Avg CLV', 'Total CLV', 'Customer Count']

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(channel_clv, x='Channel', y='Avg CLV', color='Channel',
                 title='Average CLV by Channel')
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.bar(channel_clv, x='Channel', y='Total CLV', color='Channel',
                 title='Total CLV Contribution by Channel')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# ============================================
# NEW ENHANCEMENTS - CUSTOMER RISK ANALYSIS
# (Added AFTER existing CLV functionality)
# ============================================
# ============================================

st.markdown("---")
st.markdown("## 🚨 Customer Risk Analysis Dashboard")
st.markdown("*Advanced Churn Risk Detection + Revenue Protection Strategy*")

# ============================================
# ENHANCEMENT 1: PROBABILITY ALIVE
# ============================================
with st.spinner("🔄 Computing customer risk metrics..."):
    
    # Calculate probability alive using BG/NBD model
    prob_alive_values = bgf.conditional_probability_alive(
        returning_filtered['frequency'],
        returning_filtered['recency'],
        returning_filtered['T']
    )
    
    # Convert to array and ensure between 0 and 1
    returning_filtered['Probability_Alive'] = np.array(prob_alive_values).flatten()
    returning_filtered['Probability_Alive'] = returning_filtered['Probability_Alive'].clip(0, 1)
    
    # Handle any NaN values
    returning_filtered['Probability_Alive'] = returning_filtered['Probability_Alive'].fillna(0)

# ============================================
# ENHANCEMENT 2: RISK CLASSIFICATION
# ============================================
def classify_risk(prob_alive):
    """Classify customer risk based on probability alive"""
    if pd.isna(prob_alive):
        return '⚪ Unknown'
    elif prob_alive >= 0.80:
        return '🟢 Low'
    elif prob_alive >= 0.60:
        return '🟡 Medium'
    elif prob_alive >= 0.30:
        return '🟠 High'
    else:
        return '🔴 Critical'

returning_filtered['Risk_Level'] = returning_filtered['Probability_Alive'].apply(classify_risk)

# ============================================
# ENHANCEMENT 3: MERGE FUTURE PURCHASE PREDICTIONS (90d)
# ============================================
future_90d = bgf.conditional_expected_number_of_purchases_up_to_time(
    90, returning_filtered['frequency'],
    returning_filtered['recency'], returning_filtered['T']
)
returning_filtered['Predicted_Purchases_90d'] = np.array(future_90d).flatten()

# ============================================
# ENHANCEMENT 4: PRIORITY SCORE
# ============================================
returning_filtered['CLV_Percentile'] = returning_filtered['CLV_12m'].rank(pct=True)
returning_filtered['Priority_Score'] = (
    returning_filtered['CLV_Percentile'] * 
    (1 - returning_filtered['Probability_Alive'])
)

# ============================================
# ENHANCEMENT 5: RECOMMENDED ACTION
# ============================================
def get_recommended_action(risk_level):
    """Get recommended action based on risk level"""
    action_map = {
        '🔴 Critical': '🚨 Immediate Sales Call',
        '🟠 High': '🎁 Retention Offer',
        '🟡 Medium': '📧 Email Campaign',
        '🟢 Low': '💎 Loyalty / Upsell'
    }
    return action_map.get(risk_level, '📊 Monitor')

returning_filtered['Recommended_Action'] = returning_filtered['Risk_Level'].apply(get_recommended_action)

# ============================================
# ENHANCEMENT 11: SIDEBAR FILTERS
# ============================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Risk Filters")

risk_filter_options = st.sidebar.multiselect(
    "Risk Level",
    options=['🔴 Critical', '🟠 High', '🟡 Medium', '🟢 Low'],
    default=['🔴 Critical', '🟠 High', '🟡 Medium', '🟢 Low'],
    key='risk_level_filter'
)

region_options = sorted(returning_filtered['region'].dropna().unique())
region_filter_risk = st.sidebar.multiselect(
    "Region (Risk View)",
    options=region_options,
    default=region_options,
    key='region_risk_filter'
)

channel_options = sorted(returning_filtered['channel'].dropna().unique())
channel_filter_risk = st.sidebar.multiselect(
    "Channel (Risk View)",
    options=channel_options,
    default=channel_options,
    key='channel_risk_filter'
)

clv_segment_filter = st.sidebar.multiselect(
    "CLV Segment",
    options=['🥉 Bronze', '🥈 Silver', '🥇 Gold', '💎 Platinum'],
    default=['🥉 Bronze', '🥈 Silver', '🥇 Gold', '💎 Platinum'],
    key='clv_segment_filter'
)

# Apply filters
risk_df = returning_filtered[
    (returning_filtered['Risk_Level'].isin(risk_filter_options)) &
    (returning_filtered['region'].isin(region_filter_risk)) &
    (returning_filtered['channel'].isin(channel_filter_risk)) &
    (returning_filtered['CLV_Segment'].isin(clv_segment_filter))
].copy()

# ============================================
# ENHANCEMENT 6: RISK DASHBOARD KPIs
# ============================================
st.subheader("📊 Risk Dashboard KPIs")

total_customers = len(risk_df)
high_risk_count = (risk_df['Risk_Level'] == '🟠 High').sum() if total_customers > 0 else 0
critical_count = (risk_df['Risk_Level'] == '🔴 Critical').sum() if total_customers > 0 else 0

if total_customers > 0:
    risk_at_risk_mask = risk_df['Risk_Level'].isin(['🟠 High', '🔴 Critical'])
    total_clv_at_risk = risk_df.loc[risk_at_risk_mask, 'CLV_12m'].sum()
    total_clv_overall = risk_df['CLV_12m'].sum()
else:
    total_clv_at_risk = 0
    total_clv_overall = 0

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="👥 Total Customers",
    value=f"{total_customers:,}",
    delta=f"Across {risk_df['region'].nunique()} regions" if total_customers > 0 else "No data"
)

col2.metric(
    label="🟠 High Risk Customers",
    value=f"{high_risk_count:,}",
    delta=f"{(high_risk_count/total_customers*100):.1f}% of total" if total_customers > 0 else "0%",
    delta_color="inverse"
)

col3.metric(
    label="🔴 Critical Customers",
    value=f"{critical_count:,}",
    delta=f"{(critical_count/total_customers*100):.1f}% of total" if total_customers > 0 else "0%",
    delta_color="inverse"
)

col4.metric(
    label="💸 Total CLV At Risk",
    value=f"₹{total_clv_at_risk/1e7:.2f} Cr",
    delta=f"{(total_clv_at_risk/total_clv_overall*100):.1f}% of total CLV" if total_clv_overall > 0 else "0%",
    delta_color="inverse"
)

if critical_count > 0:
    critical_clv = risk_df.loc[risk_df['Risk_Level'] == '🔴 Critical', 'CLV_12m'].sum()
    st.error(f"""
    🚨 **CRITICAL ALERT**: {critical_count} customers at critical churn risk
    representing ₹{critical_clv/1e5:.1f} L in CLV.
    **Immediate action required!**
    """)

# ============================================
# ENHANCEMENT 7: RISK VISUALIZATIONS
# ============================================
st.subheader("📈 Risk Analysis Visualizations")

if len(risk_df) == 0:
    st.warning("⚠️ No customers match the selected filters. Please adjust filters.")
else:
    viz_df = risk_df.copy()
    
    # ---- 7A: PIE CHART - Customer Count by Risk Level ----
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = viz_df['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Level', 'Count']
        
        # Order: Critical → High → Medium → Low
        risk_order = ['🔴 Critical', '🟠 High', '🟡 Medium', '🟢 Low']
        risk_counts['Risk_Level'] = pd.Categorical(
            risk_counts['Risk_Level'], 
            categories=risk_order, 
            ordered=True
        )
        risk_counts = risk_counts.sort_values('Risk_Level')
        
        fig = px.pie(
            risk_counts,
            names='Risk_Level',
            values='Count',
            title='Customer Distribution by Risk Level',
            color='Risk_Level',
            color_discrete_map={
                '🔴 Critical': '#dc2626',
                '🟠 High': '#ea580c',
                '🟡 Medium': '#ca8a04',
                '🟢 Low': '#16a34a'
            },
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- 7B: BAR CHART - Total CLV by Risk Level ----
    with col2:
        clv_by_risk = viz_df.groupby('Risk_Level', observed=True)['CLV_12m'].sum().reset_index()
        clv_by_risk['Risk_Level'] = pd.Categorical(
            clv_by_risk['Risk_Level'], 
            categories=risk_order, 
            ordered=True
        )
        clv_by_risk = clv_by_risk.sort_values('Risk_Level')
        
        fig = px.bar(
            clv_by_risk,
            x='Risk_Level',
            y='CLV_12m',
            title='Total CLV by Risk Level (₹)',
            color='Risk_Level',
            color_discrete_map={
                '🔴 Critical': '#dc2626',
                '🟠 High': '#ea580c',
                '🟡 Medium': '#ca8a04',
                '🟢 Low': '#16a34a'
            },
            text=clv_by_risk['CLV_12m'].apply(lambda x: f'₹{x/1e5:.0f}L')
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- 7C: SCATTER PLOT - Risk Quadrant Analysis ----
    st.subheader("🎯 Risk Quadrant Analysis")
    
    fig = px.scatter(
        viz_df,
        x='Probability_Alive',
        y='CLV_12m',
        size='Predicted_Purchases_90d',
        color='Risk_Level',
        hover_name='outlet_name',
        hover_data={
            'channel': True,
            'region': True,
            'Probability_Alive': False,
            'CLV_12m': False,
            'Predicted_Purchases_90d': False,
            'Risk_Level': True
        },
        color_discrete_map={
            '🔴 Critical': '#dc2626',
            '🟠 High': '#ea580c',
            '🟡 Medium': '#ca8a04',
            '🟢 Low': '#16a34a'
        },
        title='Risk Quadrant: Probability Alive vs CLV (Bubble = 90d Predicted Purchases)',
        labels={
            'Probability_Alive': 'Probability Alive',
            'CLV_12m': 'CLV (12 Months)',
            'Risk_Level': 'Risk Level'
        },
        size_max=40
    )
    
    # Add quadrant lines
    fig.add_hline(y=viz_df['CLV_12m'].median(), line_dash="dash",
                   line_color="gray", annotation_text="Median CLV")
    fig.add_vline(x=0.5, line_dash="dash",
                   line_color="gray", annotation_text="50% Probability")
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quadrant interpretation
    with st.expander("📖 How to Read This Chart"):
        st.markdown("""
        - **Top-Left** (High CLV + Low Probability Alive): **🚨 URGENT ACTION** - High-value customers at risk
        - **Top-Right** (High CLV + High Probability Alive): **💎 NURTURE** - Valuable loyal customers
        - **Bottom-Left** (Low CLV + Low Probability Alive): **⚪ DEPRIORITIZE**
        - **Bottom-Right** (Low CLV + High Probability Alive): **📈 GROWTH** - Upsell opportunity
        """)

# ============================================
# ENHANCEMENT 8: HIGH RISK CUSTOMER TABLE
# ============================================
st.markdown("---")
st.subheader("🚨 Top 25 High Risk Customers (Priority Score Ranking)")

if len(risk_df) > 0:
    
    top_25_risk = risk_df.nlargest(25, 'Priority_Score').copy()
    
    # Build display columns safely
    display_cols = []
    for col in ['outlet_name', 'channel', 'region', 'frequency', 'recency',
                'monetary_value', 'Probability_Alive', 'Predicted_Purchases_90d',
                'CLV_12m', 'Risk_Level', 'Priority_Score', 'Recommended_Action']:
        if col in top_25_risk.columns:
            display_cols.append(col)
    
    display_df = top_25_risk[display_cols].copy()
    
    # Build format dict
    format_dict = {}
    if 'monetary_value' in display_df.columns:
        format_dict['monetary_value'] = '₹{:,.0f}'
    if 'CLV_12m' in display_df.columns:
        format_dict['CLV_12m'] = '₹{:,.0f}'
    if 'Probability_Alive' in display_df.columns:
        format_dict['Probability_Alive'] = '{:.2%}'
    if 'Predicted_Purchases_90d' in display_df.columns:
        format_dict['Predicted_Purchases_90d'] = '{:.2f}'
    if 'Priority_Score' in display_df.columns:
        format_dict['Priority_Score'] = '{:.3f}'
    if 'recency' in display_df.columns:
        format_dict['recency'] = '{:.0f} days'
    
    # Apply styling
    style_df = display_df.style.format(format_dict)
    if 'Priority_Score' in display_df.columns:
        style_df = style_df.background_gradient(subset=['Priority_Score'], cmap='RdYlGn_r')
    
    st.dataframe(style_df, use_container_width=True, height=600)
    
    # Summary insights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Highest Priority Score", f"{top_25_risk['Priority_Score'].max():.3f}")
    with col2:
        st.metric("Combined CLV (Top 25)", f"₹{top_25_risk['CLV_12m'].sum()/1e5:.1f} L")
    with col3:
        st.metric("Avg Probability Alive", f"{top_25_risk['Probability_Alive'].mean():.2%}")

# ============================================
# ENHANCEMENT 9: REVENUE AT RISK TABLE
# ============================================
st.markdown("---")
st.subheader("💸 Top 20 Revenue-At-Risk Customers (High + Critical)")

if len(risk_df) > 0:
    
    revenue_at_risk = risk_df[risk_df['Risk_Level'].isin(['🟠 High', '🔴 Critical'])].copy()
    
    if len(revenue_at_risk) > 0:
        top_20_revenue_risk = revenue_at_risk.nlargest(20, 'CLV_12m').copy()
        
        rev_risk_cols = []
        for col in ['outlet_name', 'channel', 'region', 'CLV_12m',
                    'Probability_Alive', 'Risk_Level', 'Recommended_Action', 'Priority_Score']:
            if col in top_20_revenue_risk.columns:
                rev_risk_cols.append(col)
        
        revenue_risk_display = top_20_revenue_risk[rev_risk_cols].copy()
        
        # Format dict
        rev_format = {}
        if 'CLV_12m' in revenue_risk_display.columns:
            rev_format['CLV_12m'] = '₹{:,.0f}'
        if 'Probability_Alive' in revenue_risk_display.columns:
            rev_format['Probability_Alive'] = '{:.2%}'
        if 'Priority_Score' in revenue_risk_display.columns:
            rev_format['Priority_Score'] = '{:.3f}'
        
        st.dataframe(revenue_risk_display.style.format(rev_format), use_container_width=True)
        
        # Revenue at Risk Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("💸 Total Revenue At Risk", f"₹{revenue_at_risk['CLV_12m'].sum()/1e5:.1f} L")
        col2.metric("📊 Avg Revenue At Risk", f"₹{revenue_at_risk['CLV_12m'].mean():,.0f}")
        col3.metric("⚠️ Customers Affected", f"{len(revenue_at_risk):,}")
    else:
        st.success("✅ No customers in High/Critical risk categories!")

# ============================================
# ENHANCEMENT 10: DOWNLOAD BUTTON
# ============================================
st.markdown("---")
st.subheader("📥 Download Risk Analysis Report")

if len(risk_df) > 0:
    
    # Prepare full export dataframe
    export_cols = ['outlet_id']
    optional_cols = ['outlet_name', 'channel', 'region',
                     'frequency', 'recency', 'T', 'monetary_value',
                     'predicted_avg_value', 'CLV_12m', 'CLV_Segment',
                     'Probability_Alive', 'Risk_Level', 'Predicted_Purchases_90d',
                     'CLV_Percentile', 'Priority_Score', 'Recommended_Action']
    
    for col in optional_cols:
        if col in risk_df.columns:
            export_cols.append(col)
    
    export_df = risk_df[export_cols].copy()
    export_df = export_df.sort_values('Priority_Score', ascending=False)
    
    csv = export_df.to_csv(index=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download Full Risk Analysis (CSV)",
            data=csv,
            file_name=f"customer_risk_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download complete customer risk data"
        )
    
    with col2:
        if len(revenue_at_risk) > 0:
            csv_risk_cols = [c for c in export_cols if c in revenue_at_risk.columns]
            csv_risk = revenue_at_risk.sort_values('CLV_12m', ascending=False)[csv_risk_cols].to_csv(index=False)
            st.download_button(
                label="🚨 Download Revenue-At-Risk Only (CSV)",
                data=csv_risk,
                file_name=f"revenue_at_risk_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# ============================================
# STRATEGIC RECOMMENDATIONS
# ============================================
st.markdown("---")
st.subheader("💡 Strategic Recommendations")

if len(risk_df) > 0:
    
    critical_customers = risk_df[risk_df['Risk_Level'] == '🔴 Critical']
    high_customers = risk_df[risk_df['Risk_Level'] == '🟠 High']
    medium_customers = risk_df[risk_df['Risk_Level'] == '🟡 Medium']
    low_customers = risk_df[risk_df['Risk_Level'] == '🟢 Low']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(critical_customers) > 0:
            st.error(f"""
            ### 🚨 Immediate Action Required
            
            **{len(critical_customers)} Critical customers** need immediate sales calls:
            - **Revenue Impact**: ₹{critical_customers['CLV_12m'].sum()/1e5:.1f} L
            - **Avg CLV per Customer**: ₹{critical_customers['CLV_12m'].mean():,.0f}
            - **Avg Probability Alive**: {critical_customers['Probability_Alive'].mean():.2%}
            
            **Action**: Schedule sales rep visits this week
            """)
        
        if len(high_customers) > 0:
            st.warning(f"""
            ### 🎁 Retention Campaign Needed
            
            **{len(high_customers)} High-risk customers** require retention offers:
            - **Revenue Impact**: ₹{high_customers['CLV_12m'].sum()/1e5:.1f} L
            - **Avg CLV per Customer**: ₹{high_customers['CLV_12m'].mean():,.0f}
            
            **Action**: Launch 15-20% discount + personal outreach
            """)
    
    with col2:
        if len(medium_customers) > 0:
            st.info(f"""
            ### 📧 Email Campaign
            
            **{len(medium_customers)} Medium-risk customers** for email nurturing:
            - **Revenue Impact**: ₹{medium_customers['CLV_12m'].sum()/1e5:.1f} L
            
            **Action**: Personalized email series with product recommendations
            """)
        
        if len(low_customers) > 0:
            st.success(f"""
            ### 💎 Loyalty Program
            
            **{len(low_customers)} Low-risk customers** for upsell:
            - **Revenue Impact**: ₹{low_customers['CLV_12m'].sum()/1e7:.2f} Cr
            - **Loyalty potential**: HIGH
            
            **Action**: Loyalty rewards, premium SKUs, referral programs
            """)
    
    # Overall summary
    total_revenue_protected = (critical_customers['CLV_12m'].sum() + 
                                 high_customers['CLV_12m'].sum())
    
    st.markdown(f"""
    ### 📊 Executive Summary
    
    | Metric | Value |
    |--------|-------|
    | **Total Customers Analyzed** | {total_customers:,} |
    | **Total CLV (12 months)** | ₹{risk_df['CLV_12m'].sum()/1e7:.2f} Cr |
    | **Revenue At Risk (High + Critical)** | ₹{total_revenue_protected/1e7:.2f} Cr |
    | **Risk Rate** | {(len(critical_customers) + len(high_customers))/total_customers*100:.1f}% |
    | **Recommended Priority** | {'🚨 URGENT' if len(critical_customers) > 0 else '⚠️ HIGH' if len(high_customers) > 0 else '✅ NORMAL'} |
    
    **Estimated Revenue Protection Opportunity**: ₹{total_revenue_protected*0.3/1e7:.2f} Cr 
    *(assuming 30% retention success rate)*
    """)

# ============================================
# END OF ENHANCED CLV DASHBOARD
# ============================================
