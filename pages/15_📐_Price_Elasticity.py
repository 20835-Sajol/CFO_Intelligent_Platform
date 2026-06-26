import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Price Elasticity", page_icon="📐", layout="wide")
st.title("📐 Price Elasticity Analysis")
st.markdown("### Optimize Pricing & Promotions with Log-Log Regression")

data = load_data()
sales = data['sales'].merge(data['products'][['product_id','category','brand','unit_cost','mrp']], 
                              on='product_id')

# ============================================
# KPI OVERVIEW
# ============================================
st.subheader("📊 Pricing Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total SKUs", sales['product_id'].nunique())
c2.metric("Avg Selling Price", f"₹{sales['unit_price'].mean():.2f}")
c3.metric("Avg MRP", f"₹{sales['mrp'].mean():.2f}")
c4.metric("Avg Discount", f"{sales['discount_pct'].mean():.1f}%")

# ============================================
# FILTER
# ============================================
with st.expander("🎛️ Analysis Scope", expanded=True):
    f1, f2 = st.columns(2)
    target_cat = f1.selectbox("Category", ['All'] + list(sales['category'].unique()))
    target_brand = f2.selectbox("Brand", ['All'] + list(sales['brand'].unique()))
    
    elasticity_type = st.radio("Elasticity Type", ["By Product", "By Category", "By Brand"])

# ============================================
# FILTER DATA
# ============================================
df = sales.copy()
if target_cat != 'All':
    df = df[df['category'] == target_cat]
if target_brand != 'All':
    df = df[df['brand'] == target_brand]

# Aggregate by price-quantity relationship
def calculate_elasticity(group_df, group_col):
    """Calculate log-log elasticity"""
    if len(group_df) < 10:
        return None
    
    agg = group_df.groupby(group_col).agg(
        avg_price=('unit_price', 'mean'),
        total_qty=('quantity', 'sum'),
        transactions=('order_id', 'nunique')
    ).reset_index()
    
    # Remove zero/negative values for log
    agg = agg[(agg['avg_price'] > 0) & (agg['total_qty'] > 0)]
    
    if len(agg) < 5:
        return None
    
    log_price = np.log(agg['avg_price'])
    log_qty = np.log(agg['total_qty'])
    
    model = LinearRegression()
    model.fit(log_price.values.reshape(-1, 1), log_qty.values)
    
    elasticity = model.coef_[0]
    r_squared = model.score(log_price.values.reshape(-1, 1), log_qty.values)
    
    return {
        'elasticity': elasticity,
        'r_squared': r_squared,
        'data': agg
    }

# ============================================
# CALCULATE ELASTICITY
# ============================================
if elasticity_type == "By Product":
    elasticities = []
    products_to_analyze = df['product_id'].unique()[:50]  # Limit for performance
    
    for pid in products_to_analyze:
        product_df = df[df['product_id'] == pid]
        if len(product_df) > 30:
            result = calculate_elasticity(product_df, 'order_date')
            if result:
                product_name = data['products'][data['products']['product_id']==pid]['product_name'].values[0]
                elasticities.append({
                    'product_id': pid,
                    'product_name': product_name,
                    'category': product_df['category'].iloc[0],
                    'brand': product_df['brand'].iloc[0],
                    'elasticity': result['elasticity'],
                    'r_squared': result['r_squared'],
                    'avg_price': result['data']['avg_price'].mean(),
                    'total_qty': result['data']['total_qty'].sum()
                })
    
    elast_df = pd.DataFrame(elasticities)
    
elif elasticity_type == "By Category":
    elasticities = []
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 100:
            result = calculate_elasticity(cat_df, 'order_date')
            if result:
                elasticities.append({
                    'category': cat,
                    'elasticity': result['elasticity'],
                    'r_squared': result['r_squared'],
                    'avg_price': result['data']['avg_price'].mean(),
                    'total_qty': result['data']['total_qty'].sum()
                })
    elast_df = pd.DataFrame(elasticities)

elif elasticity_type == "By Brand":
    elasticities = []
    for brand in df['brand'].unique():
        brand_df = df[df['brand'] == brand]
        if len(brand_df) > 100:
            result = calculate_elasticity(brand_df, 'order_date')
            if result:
                elasticities.append({
                    'brand': brand,
                    'elasticity': result['elasticity'],
                    'r_squared': result['r_squared'],
                    'avg_price': result['data']['avg_price'].mean(),
                    'total_qty': result['data']['total_qty'].sum()
                })
    elast_df = pd.DataFrame(elasticities)

# ============================================
# INTERPRETATION
# ============================================
if elast_df is not None and len(elast_df) > 0:
    st.subheader("🎯 Elasticity Results")
    
    # Classification
    def classify(e):
        if e < -1: return '🔴 Highly Elastic'
        elif e < -0.5: return '🟡 Elastic'
        elif e < 0: return '🟢 Mildly Elastic'
        elif e == 0: return '⚪ Unit Elastic'
        else: return '🟠 Inelastic (Premium)'
    
    elast_df['classification'] = elast_df['elasticity'].apply(classify)
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Elasticity", f"{elast_df['elasticity'].mean():.3f}")
    c2.metric("Most Elastic", elast_df.loc[elast_df['elasticity'].idxmin(), 
                                            elast_df.columns[0] if 'product_name' not in elast_df.columns else 'product_name'])
    c3.metric("Least Elastic", elast_df.loc[elast_df['elasticity'].idxmax(),
                                            elast_df.columns[0] if 'product_name' not in elast_df.columns else 'product_name'])
    
    # Visualization
    c1, c2 = st.columns(2)
    
    with c1:
        name_col = 'product_name' if 'product_name' in elast_df.columns else (
                   'category' if 'category' in elast_df.columns else 'brand')
        top_n = elast_df.nsmallest(15, 'elasticity')
        
        fig = px.bar(top_n, x='elasticity', y=name_col, orientation='h',
                     color='classification', title=f'Top 15 Most Elastic {elasticity_type.split()[1]}s',
                     color_discrete_map={
                         '🔴 Highly Elastic': '#d62728',
                         '🟡 Elastic': '#ff7f0e',
                         '🟢 Mildly Elastic': '#2ca02c',
                         '🟠 Inelastic (Premium)': '#1f77b4'
                     })
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = px.scatter(elast_df, x='avg_price', y='elasticity', 
                         size='total_qty', color='classification',
                         hover_data=[name_col], title='Price vs Elasticity (bubble = volume)')
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Table
    st.dataframe(elast_df.sort_values('elasticity').style.format({
        'elasticity': '{:.3f}',
        'r_squared': '{:.3f}',
        'avg_price': '₹{:.2f}'
    }), use_container_width=True)
    
    # ============================================
    # PRICING RECOMMENDATIONS
    # ============================================
    st.markdown("---")
    st.subheader("💡 Pricing Recommendations")
    
    elastic_products = elast_df[elast_df['elasticity'] < -1]
    inelastic_products = elast_df[elast_df['elasticity'] > -0.5]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(elastic_products) > 0:
            name_col = 'product_name' if 'product_name' in elast_df.columns else (
                       'category' if 'category' in elast_df.columns else 'brand')
            st.warning(f"""
            📉 **Elastic Products (Avoid Price Increases)**
            - {len(elastic_products)} {elasticity_type.split()[1].lower()}(s) with elasticity < -1
            - **Strategy**: Price increases will significantly reduce demand
            - **Action**: Focus on volume, consider value packs
            - Examples: {', '.join(elastic_products.nlargest(3, 'elasticity')[name_col].astype(str).values)}
            """)
    
    with col2:
        if len(inelastic_products) > 0:
            name_col = 'product_name' if 'product_name' in elast_df.columns else (
                       'category' if 'category' in elast_df.columns else 'brand')
            st.success(f"""
            💎 **Inelastic Products (Pricing Power)**
            - {len(inelastic_products)} {elasticity_type.split()[1].lower()}(s) with low elasticity
            - **Strategy**: Premium pricing opportunity
            - **Action**: Test 5-10% price increases
            - Examples: {', '.join(inelastic_products.nsmallest(3, 'elasticity')[name_col].astype(str).values)}
            """)
    
    # Optimal Price Calculator
    st.markdown("---")
    st.subheader("🎯 Optimal Price Calculator (for Elastic Products)")
    
    selected_idx = st.selectbox("Select product/category", 
                                  elast_df.nsmallest(20, 'elasticity').index,
                                  format_func=lambda x: elast_df.loc[x, 
                                      'product_name' if 'product_name' in elast_df.columns else (
                                      'category' if 'category' in elast_df.columns else 'brand')])
    
    if selected_idx is not None:
        row = elast_df.loc[selected_idx]
        current_price = row['avg_price']
        elasticity = row['elasticity']
        
        # Optimal price: P* = MC * (elasticity / (1 + elasticity)) - simplified
        # For maximization: % change in price = 1/(1+elasticity)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"₹{current_price:.2f}")
        c2.metric("Elasticity", f"{elasticity:.3f}")
        c3.metric("Optimal Change", f"{1/(1+elasticity)*100:+.1f}%")
        
        # Simulate scenarios
        scenarios = [-20, -10, -5, 0, 5, 10, 20]
        sim_data = []
        for pct in scenarios:
            new_price = current_price * (1 + pct/100)
            # Quantity change = elasticity * price change %
            new_qty = row['total_qty'] * (1 + elasticity * pct/100)
            new_revenue = new_price * new_qty
            sim_data.append({
                'Price Change %': pct,
                'New Price': new_price,
                'New Quantity': new_qty,
                'New Revenue': new_revenue,
                'Revenue Change %': (new_revenue / (current_price * row['total_qty']) - 1) * 100
            })
        
        sim_df = pd.DataFrame(sim_data)
        
        fig = px.line(sim_df, x='Price Change %', y='Revenue Change %',
                      markers=True, title='Revenue Sensitivity to Price Changes')
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(sim_df.style.format({
            'New Price': '₹{:.2f}',
            'New Quantity': '{:,.0f}',
            'New Revenue': '₹{:,.0f}',
            'Revenue Change %': '{:+.2f}%'
        }), use_container_width=True)
