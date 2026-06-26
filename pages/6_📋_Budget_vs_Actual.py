import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_data
from datetime import datetime

st.set_page_config(page_title="Budget vs Actual", page_icon="📋", layout="wide")
st.title("📋 Budget vs Actual Analysis")

data = load_data()
budget = data['budget'].copy()
sales = data['sales'].copy()
expenses = data['expenses'].copy()

# ============================================
# TOP KPIs
# ============================================
st.subheader("🎯 Budget Performance Snapshot")

total_planned = budget['planned_revenue'].sum()
total_actual = budget['actual_revenue'].sum()
total_variance = total_actual - total_planned
variance_pct = (total_variance / total_planned * 100)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Planned Revenue", f"₹{total_planned/1e7:.2f} Cr")
c2.metric("Actual Revenue", f"₹{total_actual/1e7:.2f} Cr", 
          f"{variance_pct:+.2f}%")
c3.metric("Variance", f"₹{abs(total_variance)/1e7:.2f} Cr",
          "Favorable" if total_variance > 0 else "Unfavorable")
c4.metric("Achievement %", f"{total_actual/total_planned*100:.1f}%")
c5.metric("Periods Analyzed", budget['period'].nunique())

st.markdown("---")

# ============================================
# FILTER SECTION
# ============================================
with st.expander("🎛️ Filters", expanded=True):
    f1, f2, f3 = st.columns(3)
    regions_filter = f1.multiselect("Regions", budget['region'].unique(), default=budget['region'].unique())
    channels_filter = f2.multiselect("Channels", budget['channel'].unique(), default=budget['channel'].unique())
    
    date_range = f3.date_input("Period Range", 
                                value=[budget['period'].min(), budget['period'].max()],
                                min_value=budget['period'].min(),
                                max_value=budget['period'].max())

budget_f = budget[
    (budget['region'].isin(regions_filter)) &
    (budget['channel'].isin(channels_filter)) &
    (budget['period'] >= pd.Timestamp(date_range[0])) &
    (budget['period'] <= pd.Timestamp(date_range[1]))
]

# ============================================
# TREND ANALYSIS
# ============================================
st.subheader("📈 Planned vs Actual Trend")

monthly_summary = budget_f.groupby('period').agg(
    planned=('planned_revenue', 'sum'),
    actual=('actual_revenue', 'sum'),
    planned_gp_pct=('planned_gp_pct', 'mean'),
    actual_gp_pct=('actual_gp_pct', 'mean')
).reset_index()

monthly_summary['variance'] = monthly_summary['actual'] - monthly_summary['planned']
monthly_summary['variance_pct'] = (monthly_summary['variance'] / monthly_summary['planned'] * 100).round(2)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=monthly_summary['period'], y=monthly_summary['planned'],
                          mode='lines+markers', name='Planned', line=dict(color='#1f77b4', width=3)))
fig.add_trace(go.Scatter(x=monthly_summary['period'], y=monthly_summary['actual'],
                          mode='lines+markers', name='Actual', line=dict(color='#2ca02c', width=3)))
fig.add_trace(go.Bar(x=monthly_summary['period'], y=monthly_summary['variance'],
                      name='Variance', marker_color=['green' if v >= 0 else 'red' for v in monthly_summary['variance']],
                      opacity=0.3))
fig.update_layout(title='Monthly Revenue: Planned vs Actual vs Variance',
                  hovermode='x unified', template='plotly_white')
fig.update_yaxes(title_text="Revenue", secondary_y=False)
st.plotly_chart(fig, use_container_width=True)

# Achievement gauge
avg_achievement = (monthly_summary['actual'].sum() / monthly_summary['planned'].sum() * 100)
c1, c2 = st.columns([1, 2])
with c1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_achievement,
        delta={'reference': 100, 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}},
        gauge={
            'axis': {'range': [None, 120]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 80], 'color': "#ffcccb"},
                {'range': [80, 100], 'color': "#ffffcc"},
                {'range': [100, 120], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        },
        title={'text': "Achievement %"}
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.dataframe(monthly_summary[['period','planned','actual','variance','variance_pct']].style.format({
        'planned': '₹{:,.0f}',
        'actual': '₹{:,.0f}',
        'variance': '₹{:,.0f}',
        'variance_pct': '{:+.2f}%'
}), use_container_width=True)

st.markdown("---")

# ============================================
# REGION × CHANNEL HEATMAP
# ============================================
st.subheader("🔥 Variance Heatmap: Region × Channel")

region_channel_var = budget_f.pivot_table(
    index='region', columns='channel', values='variance_pct', aggfunc='mean'
).round(2)

fig = px.imshow(region_channel_var, text_auto='.1f', aspect='auto',
                color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                title='Average Variance % by Region & Channel')
st.plotly_chart(fig, use_container_width=True)

# ============================================
# TOP & BOTTOM PERFORMERS
# ============================================
st.subheader("🏆 Best & Worst Performers")

# By region
region_perf = budget_f.groupby('region').agg(
    planned=('planned_revenue', 'sum'),
    actual=('actual_revenue', 'sum')
).reset_index()
region_perf['achievement_pct'] = (region_perf['actual'] / region_perf['planned'] * 100).round(2)
region_perf = region_perf.sort_values('achievement_pct', ascending=False)

# By channel
channel_perf = budget_f.groupby('channel').agg(
    planned=('planned_revenue', 'sum'),
    actual=('actual_revenue', 'sum')
).reset_index()
channel_perf['achievement_pct'] = (channel_perf['actual'] / channel_perf['planned'] * 100).round(2)
channel_perf = channel_perf.sort_values('achievement_pct', ascending=False)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(region_perf, x='region', y='achievement_pct',
                 color='achievement_pct', color_continuous_scale='RdYlGn',
                 color_continuous_midpoint=100, title='Region Achievement %',
                 range_y=[0, 130])
    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Target")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(region_perf.style.format({
        'planned': '₹{:,.0f}', 'actual': '₹{:,.0f}', 'achievement_pct': '{:.2f}%'
    }), use_container_width=True)

with c2:
    fig = px.bar(channel_perf, x='channel', y='achievement_pct',
                 color='achievement_pct', color_continuous_scale='RdYlGn',
                 color_continuous_midpoint=100, title='Channel Achievement %',
                 range_y=[0, 130])
    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Target")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(channel_perf.style.format({
        'planned': '₹{:,.0f}', 'actual': '₹{:,.0f}', 'achievement_pct': '{:.2f}%'
    }), use_container_width=True)

st.markdown("---")

# ============================================
# GROSS MARGIN ANALYSIS
# ============================================
st.subheader("💰 Gross Margin Performance")

gp_comparison = budget_f.groupby('period').agg(
    planned_gp_pct=('planned_gp_pct', 'mean'),
    actual_gp_pct=('actual_gp_pct', 'mean')
).reset_index()
gp_comparison['gp_variance'] = (gp_comparison['actual_gp_pct'] - gp_comparison['planned_gp_pct']).round(2)

fig = go.Figure()
fig.add_trace(go.Scatter(x=gp_comparison['period'], y=gp_comparison['planned_gp_pct'],
                          mode='lines+markers', name='Planned GP %', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=gp_comparison['period'], y=gp_comparison['actual_gp_pct'],
                          mode='lines+markers', name='Actual GP %', line=dict(width=3)))
fig.update_layout(title='Gross Margin %: Planned vs Actual', template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================
# EXPENSE VARIANCE ANALYSIS
# ============================================
st.subheader("💸 Expense Variance Analysis")

expense_summary = expenses.groupby(['category', 'subcategory']).agg(
    budget=('budget', 'sum'),
    actual=('actual', 'sum')
).reset_index()
expense_summary['variance'] = expense_summary['actual'] - expense_summary['budget']
expense_summary['variance_pct'] = (expense_summary['variance'] / expense_summary['budget'] * 100).round(2)
expense_summary['status'] = expense_summary['variance_pct'].apply(
    lambda x: '🔴 Overspend' if x > 10 else ('🟡 Near Budget' if x > 0 else '🟢 Under Budget')
)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(expense_summary, x='subcategory', y='variance_pct', color='category',
                 title='Expense Variance % by Subcategory')
    fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    cat_summary = expense_summary.groupby('category').agg(
        budget=('budget', 'sum'),
        actual=('actual', 'sum')
    ).reset_index()
    cat_summary['variance_pct'] = ((cat_summary['actual'] - cat_summary['budget']) / cat_summary['budget'] * 100).round(2)
    
    fig = px.bar(cat_summary, x='category', y=['budget', 'actual'],
                 barmode='group', title='Expense Category: Budget vs Actual')
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(expense_summary.sort_values('variance_pct', ascending=False).style.format({
    'budget': '₹{:,.0f}', 'actual': '₹{:,.0f}', 
    'variance': '₹{:,.0f}', 'variance_pct': '{:+.2f}%'
}), use_container_width=True)

st.markdown("---")

# ============================================
# FORECAST vs BUDGET
# ============================================
st.subheader("🔮 Forecast vs Budget")

# Calculate run-rate from latest months
recent_months = monthly_summary.tail(3)
avg_monthly_actual = recent_months['actual'].mean()

# Project to end of period
last_period = monthly_summary['period'].max()
projected_months = pd.date_range(last_period + pd.DateOffset(months=1), periods=3, freq='MS')
projection = pd.DataFrame({
    'period': projected_months,
    'actual': [avg_monthly_actual * (1 + i*0.02) for i in range(3)]  # 2% growth
})

projection_display = pd.concat([
    monthly_summary[['period', 'actual']],
    projection
]).reset_index(drop=True)

planned_total = monthly_summary['planned'].sum()
actual_to_date = monthly_summary['actual'].sum()
projected_total = actual_to_date + projection['actual'].sum()

c1, c2, c3 = st.columns(3)
c1.metric("Plan (Full Period)", f"₹{planned_total/1e7:.2f} Cr")
c2.metric("Actual To Date", f"₹{actual_to_date/1e7:.2f} Cr")
c3.metric("Projected Total", f"₹{projected_total/1e7:.2f} Cr", 
          f"{(projected_total/planned_total*100-100):+.1f}% vs plan")

st.markdown("---")

# ============================================
# INSIGHTS & ALERTS
# ============================================
st.subheader("💡 Budget Insights & Alerts")

# Calculate insights
over_achievement = budget_f[budget_f['variance_pct'] > 10]
under_achievement = budget_f[budget_f['variance_pct'] < -10]
overspend_cats = expense_summary[expense_summary['variance_pct'] > 10]

col1, col2 = st.columns(2)

with col1:
    if len(over_achievement) > 0:
        st.success(f"""
        ✅ **Strong Performance**
        - {len(over_achievement)} period-region-channel combos exceeded plan by 10%+
        - Total over-achievement: ₹{over_achievement['actual_revenue'].sum() - over_achievement['planned_revenue'].sum():,.0f}
        """)
    
    if len(under_achievement) > 0:
        st.error(f"""
        🚨 **Underperformance Alert**
        - {len(under_achievement)} period-region-channel combos missed plan by 10%+
        - Total shortfall: ₹{abs(under_achievement['actual_revenue'].sum() - under_achievement['planned_revenue'].sum()):,.0f}
        - **Action Required**: Investigate root causes
        """)

with col2:
    if len(overspend_cats) > 0:
        st.warning(f"""
        💸 **Overspend Alert**
        - {len(overspend_cats)} expense categories over budget
        - Total overspend: ₹{overspend_cats['variance'].sum():,.0f}
        - Top offender: **{overspend_cats.iloc[0]['subcategory']}** (+{overspend_cats.iloc[0]['variance_pct']:.1f}%)
        """)

# Best/Worst month
best_month = monthly_summary.loc[monthly_summary['variance_pct'].idxmax()]
worst_month = monthly_summary.loc[monthly_summary['variance_pct'].idxmin()]

st.info(f"""
📊 **Period Highlights**
- 🏆 Best Month: **{best_month['period'].strftime('%B %Y')}** ({best_month['variance_pct']:+.1f}% variance)
- 😔 Worst Month: **{worst_month['period'].strftime('%B %Y')}** ({worst_month['variance_pct']:+.1f}% variance)
- 📈 Trend: {'Improving ↗️' if monthly_summary['variance_pct'].iloc[-1] > monthly_summary['variance_pct'].iloc[0] else 'Declining ↘️'}
""")

# ============================================
# DOWNLOAD SECTION
# ============================================
st.markdown("---")
st.subheader("📥 Download Budget Reports")

c1, c2, c3 = st.columns(3)
with c1:
    csv = monthly_summary.to_csv(index=False)
    st.download_button("📊 Monthly Summary", csv, "budget_monthly.csv", "text/csv")
with c2:
    csv = budget_f.to_csv(index=False)
    st.download_button("📋 Detailed Data", csv, "budget_detailed.csv", "text/csv")
with c3:
    csv = expense_summary.to_csv(index=False)
    st.download_button("💸 Expense Report", expense_summary.to_csv(index=False), 
                       "expense_variance.csv", "text/csv")
