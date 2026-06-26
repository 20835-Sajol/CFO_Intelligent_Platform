import streamlit as st
import plotly.express as px
from utils.data_loader import load_data
from utils.visualizations import plot_bar

st.set_page_config(page_title="Profitability", page_icon="💰", layout="wide")
st.title("💰 Profitability Analysis")

data = load_data()
sales = data['sales'].merge(data['products'][['product_id', 'category', 'brand']], on='product_id')

# KPIs
sales['gross_margin_pct'] = (sales['gross_profit'] / sales['net_amount']) * 100
total_rev = sales['net_amount'].sum()
total_gp = sales['gross_profit'].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue", f"₹{total_rev/1e7:.2f} Cr")
k2.metric("COGS", f"₹{sales['cogs'].sum()/1e7:.2f} Cr")
k3.metric("Gross Profit", f"₹{total_gp/1e7:.2f} Cr")
k4.metric("Gross Margin", f"{total_gp/total_rev*100:.2f}%")

st.markdown("---")

# Margin by dimension
dim = st.selectbox("Analyze Margin by", ['category', 'brand', 'channel', 'region'])

margin_df = sales.groupby(dim).agg(
    revenue=('net_amount', 'sum'),
    cogs=('cogs', 'sum'),
    gross_profit=('gross_profit', 'sum')
).reset_index()
margin_df['margin_pct'] = (margin_df['gross_profit'] / margin_df['revenue']) * 100
margin_df = margin_df.sort_values('margin_pct', ascending=False)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(margin_df, x=dim, y='revenue', color='margin_pct',
                 color_continuous_scale='RdYlGn', title=f'Revenue & Margin by {dim.title()}')
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.bar(margin_df, x=dim, y='margin_pct', color='margin_pct',
                 color_continuous_scale='RdYlGn', title=f'Margin % by {dim.title()}')
    st.plotly_chart(fig, use_container_width=True)

# Top/Bottom performers
st.subheader("🏆 Top & Bottom Performers")
c1, c2 = st.columns(2)
with c1:
    st.write("**Top 10 by Margin**")
    st.dataframe(margin_df.head(10).style.format({'revenue': '₹{:,.0f}', 'margin_pct': '{:.2f}%'}), use_container_width=True)
with c2:
    st.write("**Bottom 10 by Margin**")
    st.dataframe(margin_df.tail(10).style.format({'revenue': '₹{:,.0f}', 'margin_pct': '{:.2f}%'}), use_container_width=True)

# P&L Waterfall (simulated)
st.subheader("📊 Monthly P&L Bridge")
sales['month'] = sales['order_date'].dt.to_period('M').astype(str)
monthly = sales.groupby('month').agg(rev=('net_amount','sum'), cogs=('cogs','sum')).reset_index()
monthly['gross_profit'] = monthly['rev'] - monthly['cogs']

# Add simulated expenses
monthly['marketing'] = monthly['rev'] * 0.12
monthly['logistics'] = monthly['rev'] * 0.08
monthly['admin'] = monthly['rev'] * 0.05
monthly['ebitda'] = monthly['gross_profit'] - monthly['marketing'] - monthly['logistics'] - monthly['admin']

st.dataframe(monthly.style.format({
    'rev': '₹{:,.0f}', 'cogs': '₹{:,.0f}', 'gross_profit': '₹{:,.0f}',
    'marketing': '₹{:,.0f}', 'logistics': '₹{:,.0f}', 'admin': '₹{:,.0f}', 'ebitda': '₹{:,.0f}'
}), use_container_width=True)
