import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data

st.set_page_config(page_title="AI/ML Module", page_icon="🤖", layout="wide")
st.title("🤖 AI/ML Intelligent Module")

data = load_data()
sales = data['sales'].merge(data['products'][['product_id','product_name','category','brand']], on='product_id')

tab1, tab2, tab3 = st.tabs(["👥 RFM Segmentation", "🛒 Market Basket", "🎯 Recommendations"])

# ============================================
# TAB 1: RFM SEGMENTATION
# ============================================
with tab1:
    st.subheader("👥 Customer Segmentation using RFM + K-Means")
    
    snapshot_date = sales['order_date'].max() + pd.Timedelta(days=1)
    
    rfm = sales.groupby('outlet_id').agg(
        recency=('order_date', lambda x: (snapshot_date - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('net_amount', 'sum')
    ).reset_index()
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", len(rfm))
    c2.metric("Avg Recency (days)", f"{rfm['recency'].mean():.0f}")
    c3.metric("Avg Frequency", f"{rfm['frequency'].mean():.1f}")
    c4.metric("Avg Monetary", f"₹{rfm['monetary'].mean():,.0f}")
    
    # K-Means clustering
    n_clusters = st.slider("Number of Segments", 3, 8, 5)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency','frequency','monetary']])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['segment'] = kmeans.fit_predict(rfm_scaled)
    
    # Auto-label segments
    seg_summary = rfm.groupby('segment').agg(
        recency=('recency','mean'),
        frequency=('frequency','mean'),
        monetary=('monetary','mean'),
        count=('outlet_id','count')
    ).reset_index()
    
    def label_segment(row):
        if row['monetary'] > seg_summary['monetary'].quantile(0.75) and row['frequency'] > seg_summary['frequency'].median():
            return '🌟 Champions'
        elif row['recency'] > seg_summary['recency'].quantile(0.75):
            return '😴 At Risk'
        elif row['frequency'] < seg_summary['frequency'].quantile(0.25):
            return '🆕 New Customers'
        elif row['monetary'] > seg_summary['monetary'].median():
            return '💎 Loyal'
        else:
            return '⚪ Regular'
    
    seg_labels = {row['segment']: label_segment(row) for _, row in seg_summary.iterrows()}
    rfm['segment_label'] = rfm['segment'].map(seg_labels)
    
    # 3D scatter
    fig = px.scatter_3d(rfm, x='recency', y='frequency', z='monetary',
                        color='segment_label', hover_data=['outlet_id'],
                        title=f'Customer Segments ({n_clusters} clusters)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment summary table
    seg_display = rfm.groupby('segment_label').agg(
        customers=('outlet_id','count'),
        avg_recency=('recency','mean'),
        avg_frequency=('frequency','mean'),
        avg_monetary=('monetary','mean'),
        total_revenue=('monetary','sum')
    ).reset_index().sort_values('total_revenue', ascending=False)
    
    st.dataframe(seg_display.style.format({
        'avg_recency': '{:.0f} days',
        'avg_frequency': '{:.1f}',
        'avg_monetary': '₹{:,.0f}',
        'total_revenue': '₹{:,.0f}'
    }), use_container_width=True)
    
    # Recommendation per segment
    st.subheader("💡 Recommended Action per Segment")
    actions = {
        '🌟 Champions': '🎁 Reward with exclusive offers, early access to new products',
        '😴 At Risk': '📧 Re-engagement campaign, special discounts, personal outreach',
        '🆕 New Customers': '👋 Onboarding journey, welcome offers, education content',
        '💎 Loyal': '💎 Loyalty program, premium SKUs, cross-sell opportunities',
        '⚪ Regular': '📈 Upsell campaigns, bundle offers, frequency rewards'
    }
    for _, row in seg_display.iterrows():
        with st.expander(f"{row['segment_label']} - {row['customers']} customers"):
            st.write(f"**Total Revenue:** ₹{row['total_revenue']:,.0f}")
            st.write(f"**Recommendation:** {actions.get(row['segment_label'], 'Custom strategy')}")

# ============================================
# TAB 2: MARKET BASKET ANALYSIS
# ============================================
with tab2:
    st.subheader("🛒 Market Basket Analysis (Apriori)")
    
    min_support = st.slider("Minimum Support", 0.001, 0.05, 0.005, 0.001)
    
    # Create basket
    basket = sales.groupby(['order_id', 'category'])['quantity'].sum().unstack(fill_value=0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    with st.spinner("Running Apriori algorithm..."):
        frequent = apriori(basket, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent, metric='lift', min_threshold=1.0)
    
    st.success(f"✅ Found {len(rules)} association rules")
    
    if len(rules) > 0:
        rules_display = rules.sort_values('lift', ascending=False).head(20)
        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
        
        fig = px.scatter(rules_display, x='support', y='confidence', 
                         size='lift', color='lift', hover_data=['antecedents','consequents'],
                         title='Association Rules - Support vs Confidence (size=Lift)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(rules_display[['antecedents','consequents','support','confidence','lift']].head(20), use_container_width=True)

# ============================================
# TAB 3: PRODUCT RECOMMENDATIONS
# ============================================
with tab3:
    st.subheader("🎯 Product Recommendation Engine")
    
    # Co-occurrence based
    product_pivot = sales.groupby(['order_id','product_id'])['quantity'].sum().unstack(fill_value=0)
    product_pivot = product_pivot.applymap(lambda x: 1 if x > 0 else 0)
    
    # Product similarity using cosine
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(product_pivot.T)
    similarity_df = pd.DataFrame(similarity, 
                                  index=product_pivot.columns, 
                                  columns=product_pivot.columns)
    
    selected_product = st.selectbox("Select a Product", 
                                     data['products']['product_name'].sample(20).values)
    
    pid = data['products'][data['products']['product_name']==selected_product]['product_id'].values[0]
    
    if pid in similarity_df.index:
        similar = similarity_df[pid].sort_values(ascending=False).head(11)[1:]
        similar_df = pd.DataFrame({
            'product_id': similar.index,
            'similarity_score': similar.values
        }).merge(data['products'][['product_id','product_name','category']], on='product_id')
        
        st.write(f"### Products similar to **{selected_product}**:")
        st.dataframe(similar_df, use_container_width=True)
        
        fig = px.bar(similar_df.head(10), x='similarity_score', y='product_name',
                     orientation='h', color='similarity_score',
                     title='Top 10 Similar Products (Cosine Similarity)')
        st.plotly_chart(fig, use_container_width=True)
