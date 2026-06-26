import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data
from collections import Counter

st.set_page_config(page_title="Graph Analytics", page_icon="🕸️", layout="wide")
st.title("🕸️ Distribution Network Analytics")

try:
    import networkx as nx
except ImportError:
    st.error("Install networkx: `pip install networkx`")
    st.stop()

data = load_data()
sales = data['sales']

# ============================================
# BUILD GRAPH: Outlet → Product Purchase Patterns
# ============================================
st.subheader("🔗 Build Distribution Network")

graph_type = st.selectbox("Graph Type", [
    "Outlet-Product Bipartite Network",
    "Product Co-purchase Network",
    "Region-Channel Network"
])

if graph_type == "Outlet-Product Bipartite Network":
    # Build bipartite graph
    G = nx.Graph()
    
    # Sample for performance
    sample_sales = sales.sample(min(5000, len(sales)), random_state=42)
    
    for _, row in sample_sales.iterrows():
        outlet = f"O_{row['outlet_id']}"
        product = f"P_{row['product_id']}"
        G.add_edge(outlet, product, weight=row['quantity'])
    
    # Add attributes
    outlets = data['outlets'].set_index('outlet_id')
    for outlet_id in sample_sales['outlet_id'].unique():
        if f"O_{outlet_id}" in G.nodes:
            if outlet_id in outlets.index:
                G.nodes[f"O_{outlet_id}"]['type'] = 'outlet'
                G.nodes[f"O_{outlet_id}"]['region'] = outlets.loc[outlet_id, 'region']
                G.nodes[f"O_{outlet_id}"]['channel'] = outlets.loc[outlet_id, 'outlet_type']
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Nodes", G.number_of_nodes())
    c2.metric("Total Edges", G.number_of_edges())
    c3.metric("Density", f"{nx.density(G):.4f}")
    c4.metric("Connected Components", nx.number_connected_components(G))
    
    # Top outlets by centrality
    st.subheader("🏆 Most Connected Outlets (Degree Centrality)")
    degree_cent = nx.degree_centrality(G)
    top_outlets = sorted([(k, v) for k, v in degree_cent.items() if k.startswith('O_')], 
                          key=lambda x: x[1], reverse=True)[:20]
    
    top_df = pd.DataFrame(top_outlets, columns=['Outlet', 'Centrality'])
    top_df['outlet_id'] = top_df['Outlet'].str.replace('O_', '')
    top_df = top_df.merge(data['outlets'][['outlet_id','outlet_name','outlet_type']], on='outlet_id')
    # Merge region from sales data
    outlet_regions = data['sales'][['outlet_id', 'region']].drop_duplicates()
    top_df = top_df.merge(outlet_regions, on='outlet_id', how='left')
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(top_df[['outlet_name','region','outlet_type','Centrality']].head(15), use_container_width=True)
    
    with c2:
        fig = px.bar(top_df.head(15), x='Centrality', y='outlet_name', orientation='h',
                     color='Centrality', title='Top 15 Outlets by Degree Centrality')
        st.plotly_chart(fig, use_container_width=True)

elif graph_type == "Product Co-purchase Network":
    st.markdown("### Products Frequently Bought Together")
    
    # Build product co-purchase graph
    G = nx.Graph()
    sample_orders = sales['order_id'].unique()[:3000]
    sample_sales = sales[sales['order_id'].isin(sample_orders)]
    
    for order_id, group in sample_sales.groupby('order_id'):
        products = group['product_id'].unique()
        for i, p1 in enumerate(products):
            for p2 in products[i+1:]:
                if G.has_edge(p1, p2):
                    G[p1][p2]['weight'] += 1
                else:
                    G.add_edge(p1, p2, weight=1)
    
    # Remove low-frequency edges
    edges_to_remove = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] < 5]
    G.remove_edges_from(edges_to_remove)
    
    # Largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc)
    
    # Community detection
    communities = nx.community.greedy_modularity_communities(G_sub)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", G_sub.number_of_nodes())
    c2.metric("Co-purchases", G_sub.number_of_edges())
    c3.metric("Communities", len(communities))
    c4.metric("Modularity", f"{nx.community.modularity(G_sub, communities):.3f}")
    
    # Most central products
    st.subheader("🎯 Most Important Products (PageRank)")
    pagerank = nx.pagerank(G_sub)
    top_products = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
    
    top_df = pd.DataFrame(top_products, columns=['product_id', 'PageRank'])
    top_df = top_df.merge(data['products'][['product_id','product_name','category']], on='product_id')
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(top_df[['product_name','category','PageRank']].head(15), use_container_width=True)
    with c2:
        fig = px.bar(top_df.head(15), x='PageRank', y='product_name', orientation='h',
                     color='category', title='Top Products by PageRank (Co-purchase Importance)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Community summary
    st.subheader("👥 Product Communities")
    comm_data = []
    for i, comm in enumerate(communities):
        comm_products = [data['products'][data['products']['product_id']==p]['product_name'].values[0] 
                         for p in list(comm)[:5]]
        comm_data.append({
            'Community': f'Group {i+1}',
            'Size': len(comm),
            'Top Products': ', '.join(comm_products)
        })
    
    st.dataframe(pd.DataFrame(comm_data), use_container_width=True)

else:  # Region-Channel
    G = nx.Graph()
    region_channel = sales.groupby(['region', 'channel'])['net_amount'].sum().reset_index()
    
    for _, row in region_channel.iterrows():
        G.add_edge(f"R_{row['region']}", f"C_{row['channel']}", weight=row['net_amount'])
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Build plotly visualization
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=np.log1p(edge[2]['weight'])/2, color='gray'),
            hoverinfo='none'
        ))
    
    node_trace_regions = go.Scatter(
        x=[pos[n][0] for n in G.nodes() if n.startswith('R_')],
        y=[pos[n][1] for n in G.nodes() if n.startswith('R_')],
        mode='markers+text',
        marker=dict(size=30, color='lightblue'),
        text=[n.replace('R_', '') for n in G.nodes() if n.startswith('R_')],
        textposition='top center',
        name='Region'
    )
    
    node_trace_channels = go.Scatter(
        x=[pos[n][0] for n in G.nodes() if n.startswith('C_')],
        y=[pos[n][1] for n in G.nodes() if n.startswith('C_')],
        mode='markers+text',
        marker=dict(size=25, color='lightcoral'),
        text=[n.replace('C_', '') for n in G.nodes() if n.startswith('C_')],
        textposition='top center',
        name='Channel'
    )
    
    fig = go.Figure(data=edge_trace + [node_trace_regions, node_trace_channels])
    fig.update_layout(title='Region-Channel Network', showlegend=True, 
                      hovermode='closest', height=600)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# INSIGHTS
# ============================================
st.markdown("---")
st.subheader("💡 Network Insights")

st.info("""
**Why Graph Analytics for FMCG CFO:**
- 🔍 **Identify Key Hubs**: Outlets/products that drive distribution
- 🎯 **Cross-sell Opportunities**: Products frequently bought together
- 🌐 **Network Resilience**: Critical nodes for business continuity
- 👥 **Community Detection**: Natural product/outlet groupings
- 📊 **Centrality Analysis**: Influence & importance ranking
- 🔗 **Distribution Optimization**: Streamline supply chain
""")
