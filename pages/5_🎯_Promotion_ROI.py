import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_data

st.set_page_config(page_title="Promotion ROI", page_icon="🎯", layout="wide")
st.title("🎯 Promotion ROI Analysis")

data = load_data()
sales = data['sales']
schemes = data['schemes']
products = data['products']

# Merge sales with product info
sales = sales.merge(products[['product_id', 'category', 'brand']], on='product_id', how='left')

# ============================================
# TOP KPIs
# ============================================
st.subheader("📊 Promotion Overview")

promo_sales = sales[sales['has_promotion'] == True]
non_promo_sales = sales[sales['has_promotion'] == False]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Promo Sales", f"₹{promo_sales['net_amount'].sum()/1e7:.2f} Cr")
c2.metric("Promo Revenue %", f"{promo_sales['net_amount'].sum()/sales['net_amount'].sum()*100:.1f}%")
c3.metric("Avg Discount (Promo)", f"{promo_sales['discount_pct'].mean():.1f}%")
c4.metric("Active Schemes", schemes['scheme_id'].nunique())
c5.metric("Total Promo Budget", f"₹{schemes['budget'].sum()/1e7:.2f} Cr")

st.markdown("---")

# ============================================
# PROMO vs NON-PROMO COMPARISON
# ============================================
st.subheader("⚖️ Promo vs Non-Promo Performance")

comparison = pd.DataFrame({
    'Metric': ['Avg Order Value', 'Avg Quantity', 'Avg Revenue/Order', 'Avg Discount %'],
    'With Promotion': [
        promo_sales.groupby('order_id')['net_amount'].sum().mean(),
        promo_sales.groupby('order_id')['quantity'].sum().mean(),
        promo_sales['net_amount'].mean(),
        promo_sales['discount_pct'].mean()
    ],
    'Without Promotion': [
        non_promo_sales.groupby('order_id')['net_amount'].sum().mean(),
        non_promo_sales.groupby('order_id')['quantity'].sum().mean(),
        non_promo_sales['net_amount'].mean(),
        non_promo_sales['discount_pct'].mean()
    ]
})
comparison['Lift %'] = ((comparison['With Promotion'] - comparison['Without Promotion']) / 
                          comparison['Without Promotion'] * 100).round(2)

st.dataframe(comparison.style.format({
    'With Promotion': '₹{:,.2f}',
    'Without Promotion': '₹{:,.2f}',
    'Lift %': '{:+.2f}%'
}, na_rep="-"), use_container_width=True)

# ============================================
# SCHEME PERFORMANCE
# ============================================
st.subheader("🏆 Scheme Performance")

scheme_perf = sales[sales['has_promotion']==True].groupby('scheme_id').agg(
    revenue=('net_amount', 'sum'),
    orders=('order_id', 'nunique'),
    quantity=('quantity', 'sum'),
    discount_given=('discount_amount', 'sum'),
    gross_profit=('gross_profit', 'sum')
).reset_index()

scheme_perf = scheme_perf.merge(schemes[['scheme_id', 'scheme_name', 'budget', 'discount_pct']], 
                                  on='scheme_id', how='left')
scheme_perf['roi_pct'] = ((scheme_perf['gross_profit'] - scheme_perf['budget']) / scheme_perf['budget'] * 100).round(2)
scheme_perf['revenue_per_rupee'] = (scheme_perf['revenue'] / scheme_perf['budget']).round(2)

scheme_perf_display = scheme_perf.sort_values('roi_pct', ascending=False)

# Visualize
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(scheme_perf_display, x='scheme_name', y='revenue',
                 color='roi_pct', color_continuous_scale='RdYlGn',
                 title='Revenue by Scheme (colored by ROI %)')
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.scatter(scheme_perf_display, x='budget', y='revenue', 
                     size='quantity', color='roi_pct',
                     hover_data=['scheme_name'], 
                     color_continuous_scale='RdYlGn',
                     title='Budget vs Revenue (Bubble = Volume)')
    fig.add_trace(go.Scatter(x=[scheme_perf_display['budget'].min(), scheme_perf_display['budget'].max()],
                              y=[scheme_perf_display['budget'].min(), scheme_perf_display['budget'].max()],
                              mode='lines', name='Break-even', line=dict(dash='dash', color='red')))
    st.plotly_chart(fig, use_container_width=True)

# Top performers table
st.write("**📋 Scheme Details:**")
st.dataframe(scheme_perf_display[['scheme_name','budget','revenue','gross_profit','roi_pct','revenue_per_rupee']].style.format({
    'budget': '₹{:,.0f}',
    'revenue': '₹{:,.0f}',
    'gross_profit': '₹{:,.0f}',
    'roi_pct': '{:+.2f}%',
    'revenue_per_rupee': '{:.2f}x'
}), use_container_width=True)

st.markdown("---")

# ============================================
# TIME-BASED PROMO ANALYSIS
# ============================================
st.subheader("📅 Promotion Impact Over Time")

sales['month'] = sales['order_date'].dt.to_period('M').astype(str)
monthly_promo = sales.groupby(['month', 'has_promotion'])['net_amount'].sum().reset_index()
monthly_promo['has_promotion'] = monthly_promo['has_promotion'].map({True: 'Promo', False: 'Non-Promo'})

fig = px.line(monthly_promo, x='month', y='net_amount', color='has_promotion',
              markers=True, title='Monthly Revenue: Promo vs Non-Promo')
st.plotly_chart(fig, use_container_width=True)

# ============================================
# CHANNEL-LEVEL PROMO EFFECTIVENESS
# ============================================
st.subheader("🏪 Channel-wise Promotion Effectiveness")

channel_promo = sales.groupby(['channel', 'has_promotion']).agg(
    revenue=('net_amount', 'sum'),
    orders=('order_id', 'nunique'),
    avg_discount=('discount_pct', 'mean')
).reset_index()

channel_pivot = channel_promo.pivot(index='channel', columns='has_promotion', values='revenue').reset_index()
channel_pivot.columns = ['Channel', 'Non_Promo_Revenue', 'Promo_Revenue']
channel_pivot = channel_pivot.fillna(0)
channel_pivot['Promo_Share_%'] = (channel_pivot['Promo_Revenue'] / 
                                   (channel_pivot['Promo_Revenue'] + channel_pivot['Non_Promo_Revenue']) * 100).round(2)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(channel_pivot.melt(id_vars='Channel'), 
                 x='Channel', y='value', color='variable',
                 barmode='group', title='Channel Revenue: Promo vs Non-Promo')
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.pie(channel_pivot, names='Channel', values='Promo_Revenue',
                 title='Promo Revenue Share by Channel', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(channel_pivot, use_container_width=True)

st.markdown("---")

# ============================================
# CATEGORY-LEVEL PROMO ROI
# ============================================
st.subheader("📦 Category-wise Promo Performance")

cat_promo = sales.groupby('category').apply(lambda x: pd.Series({
    'total_revenue': x['net_amount'].sum(),
    'promo_revenue': x[x['has_promotion']]['net_amount'].sum(),
    'non_promo_revenue': x[~x['has_promotion']]['net_amount'].sum(),
    'avg_margin_promo': (x[x['has_promotion']]['gross_profit'].sum() / 
                          x[x['has_promotion']]['net_amount'].sum() * 100),
    'avg_margin_non_promo': (x[~x['has_promotion']]['gross_profit'].sum() / 
                              x[~x['has_promotion']]['net_amount'].sum() * 100),
    'promo_orders': x[x['has_promotion']]['order_id'].nunique()
})).reset_index()

cat_promo['promo_share_pct'] = (cat_promo['promo_revenue'] / cat_promo['total_revenue'] * 100).round(2)
cat_promo['margin_impact'] = (cat_promo['avg_margin_promo'] - cat_promo['avg_margin_non_promo']).round(2)

fig = px.scatter(cat_promo, x='promo_share_pct', y='margin_impact',
                 size='total_revenue', color='margin_impact',
                 hover_data=['category'],
                 color_continuous_scale='RdYlGn',
                 title='Category Position: Promo Dependency vs Margin Impact')
fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Margin Neutral")
fig.add_vline(x=50, line_dash="dash", line_color="blue", annotation_text="50% Promo Dependency")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(cat_promo.style.format({
    'total_revenue': '₹{:,.0f}',
    'promo_revenue': '₹{:,.0f}',
    'non_promo_revenue': '₹{:,.0f}',
    'avg_margin_promo': '{:.2f}%',
    'avg_margin_non_promo': '{:.2f}%',
    'margin_impact': '{:+.2f} pp',
    'promo_share_pct': '{:.2f}%'
}), use_container_width=True)

st.markdown("---")

# ============================================
# PROMO INSIGHTS & RECOMMENDATIONS
# ============================================
st.subheader("💡 AI-Powered Promo Insights")

# Calculate insights
best_scheme = scheme_perf_display.iloc[0]
worst_scheme = scheme_perf_display.iloc[-1]
high_promo_cat = cat_promo.sort_values('promo_share_pct', ascending=False).iloc[0]
negative_margin_cats = cat_promo[cat_promo['margin_impact'] < -2]

col1, col2 = st.columns(2)

with col1:
    st.success(f"""
    ✅ **Best Performing Scheme**: {best_scheme['scheme_name']}
    - ROI: **{best_scheme['roi_pct']:.1f}%**
    - Revenue: ₹{best_scheme['revenue']/1e5:.1f} L
    - Revenue per ₹ spent: {best_scheme['revenue_per_rupee']:.2f}x
    """)

with col2:
    st.error(f"""
    ⚠️ **Underperforming Scheme**: {worst_scheme['scheme_name']}
    - ROI: **{worst_scheme['roi_pct']:.1f}%**
    - Consider revising or discontinuing
    """)

st.warning(f"""
🎯 **High Promo Dependency**: **{high_promo_cat['category']}** 
- {high_promo_cat['promo_share_pct']:.1f}% of revenue from promotions
- Risk: Margin erosion ({high_promo_cat['margin_impact']:.1f} pp impact)
- **Recommendation**: Reduce discount %, focus on value-added promotions
""")

if len(negative_margin_cats) > 0:
    st.error(f"""
🚨 **Margin Alert**: {len(negative_margin_cats)} categories have negative margin impact from promotions:
- {', '.join(negative_margin_cats['category'].values)}
- **Action**: Review discount strategy, consider non-discount promotions (BOGO, samples)
""")
