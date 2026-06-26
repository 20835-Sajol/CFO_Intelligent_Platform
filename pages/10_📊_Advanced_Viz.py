import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data

st.set_page_config(page_title="Advanced Visualizations", page_icon="📊", layout="wide")
st.title("📊 Advanced Visualizations")

data = load_data()
sales = data['sales'].merge(data['products'][['product_id','category','brand']], on='product_id')
sales = sales.merge(data['outlets'][['outlet_id','outlet_type']], on='outlet_id', suffixes=('','_outlet'))

viz_type = st.selectbox("Select Visualization", 
                         ["🌳 Treemap", "☀️ Sunburst", "🌊 Sankey", "🔥 Calendar Heatmap", "📈 Cohort Analysis"])

# ============================================
# TREEMAP
# ============================================
if viz_type == "🌳 Treemap":
    st.subheader("🌳 Hierarchical Treemap: Region → Channel → Category")
    
    hier = sales.groupby(['region','channel','category'])['net_amount'].sum().reset_index()
    
    fig = px.treemap(hier, path=['region','channel','category'], values='net_amount',
                     color='net_amount', color_continuous_scale='Viridis',
                     title='Revenue Hierarchy')
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# SUNBURST
# ============================================
elif viz_type == "☀️ Sunburst":
    st.subheader("☀️ Sunburst: Brand → Category → Subcategory")
    
    products_full = sales.merge(data['products'][['product_id','subcategory']], on='product_id')
    hier = products_full.groupby(['brand','category','subcategory'])['net_amount'].sum().reset_index()
    
    fig = px.sunburst(hier, path=['brand','category','subcategory'], values='net_amount',
                      color='net_amount', color_continuous_scale='RdBu',
                      title='Brand → Category → Subcategory Revenue')
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# SANKEY
# ============================================
elif viz_type == "🌊 Sankey":
    st.subheader("🌊 Sankey Flow: Region → Channel → Category")
    
    # Build source/target lists
    flows = sales.groupby(['region','channel','category'])['net_amount'].sum().reset_index()
    flows = flows[flows['net_amount'] > 0]
    
    regions = flows['region'].unique().tolist()
    channels = flows['channel'].unique().tolist()
    categories = flows['category'].unique().tolist()
    
    all_nodes = regions + channels + categories
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    
    sources, targets, values = [], [], []
    # Region -> Channel
    rc = flows.groupby(['region','channel'])['net_amount'].sum().reset_index()
    for _, r in rc.iterrows():
        sources.append(node_idx[r['region']])
        targets.append(node_idx[r['channel']])
        values.append(r['net_amount'])
    
    # Channel -> Category
    cc = flows.groupby(['channel','category'])['net_amount'].sum().reset_index()
    for _, r in cc.iterrows():
        sources.append(node_idx[r['channel']])
        targets.append(node_idx[r['category']])
        values.append(r['net_amount'])
    
    fig = go.Figure(go.Sankey(
        node=dict(pad=20, thickness=25, label=all_nodes, color='steelblue'),
        link=dict(source=sources, target=targets, value=values)
    ))
    fig.update_layout(title_text="Revenue Flow: Region → Channel → Category", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# CALENDAR HEATMAP
# ============================================
elif viz_type == "🔥 Calendar Heatmap":
    st.subheader("🔥 Daily Revenue Calendar")
    
    daily = sales.groupby('order_date')['net_amount'].sum().reset_index()
    daily['year'] = daily['order_date'].dt.year
    daily['month'] = daily['order_date'].dt.month
    daily['day'] = daily['order_date'].dt.day
    daily['weekday'] = daily['order_date'].dt.dayofweek
    
    year = st.selectbox("Year", sorted(daily['year'].unique()))
    daily_yr = daily[daily['year'] == year]
    
    fig = px.density_heatmap(daily_yr, x='month', y='weekday', z='net_amount',
                              nbinsx=12, color_continuous_scale='YlOrRd',
                              title=f'Daily Revenue Heatmap {year}')
    fig.update_yaxes(tickmode='array', tickvals=list(range(7)),
                     ticktext=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# COHORT ANALYSIS
# ============================================
elif viz_type == "📈 Cohort Analysis":
    st.subheader("📈 Customer Retention Cohort")
    
    # First purchase month per outlet
    first_purchase = sales.groupby('outlet_id')['order_date'].min().reset_index()
    first_purchase['cohort'] = first_purchase['order_date'].dt.to_period('M')
    
    sales_co = sales.merge(first_purchase[['outlet_id','cohort']], on='outlet_id')
    sales_co['order_period'] = sales_co['order_date'].dt.to_period('M')
    sales_co['period_number'] = (sales_co['order_period'] - sales_co['cohort']).apply(lambda x: x.n)
    
    # Cohort pivot
    cohort_data = sales_co.groupby(['cohort','period_number'])['outlet_id'].nunique().reset_index()
    cohort_table = cohort_data.pivot(index='cohort', columns='period_number', values='outlet_id')
    cohort_size = cohort_table.iloc[:, 0]
    retention = cohort_table.divide(cohort_size, axis=0) * 100
    
    # ⬇️ ADD THIS EXACT LINE TO FIX THE ERROR ⬇️
    retention.index = retention.index.astype(str)
    
    fig = px.imshow(retention, text_auto='.1f', aspect='auto',
                    color_continuous_scale='YlGnBu',
                    title='Customer Retention Cohort (% Active)')
    st.plotly_chart(fig, use_container_width=True)