import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent color palette
COLORS = {
    'primary': '#1f77b4',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8'
}

def plot_trend(df, x, y, title, color=COLORS['primary']):
    fig = px.line(df, x=x, y=y, title=title, markers=True)
    fig.update_traces(line_color=color, line_width=3)
    fig.update_layout(hovermode='x unified', template='plotly_white')
    return fig

def plot_bar(df, x, y, title, color_col=None, orientation='v'):
    fig = px.bar(df, x=x, y=y, title=title, color=color_col, orientation=orientation)
    fig.update_layout(template='plotly_white')
    return fig

def plot_pie(df, names, values, title):
    fig = px.pie(df, names=names, values=values, title=title, hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_heatmap(df, x, y, values, title):
    pivot = df.pivot_table(index=y, columns=x, values=values, aggfunc='sum').fillna(0)
    fig = px.imshow(pivot, title=title, aspect='auto', color_continuous_scale='RdYlGn')
    return fig

def kpi_card(label, value, delta=None):
    """HTML for KPI card"""
    delta_color = "green" if delta and delta > 0 else "red"
    delta_html = f"<span style='color:{delta_color};font-size:14px'>{'+' if delta > 0 else ''}{delta:.1f}%</span>" if delta else ""
    return f"""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                padding:20px;border-radius:10px;color:white;margin:5px">
        <div style="font-size:13px;opacity:0.9">{label}</div>
        <div style="font-size:28px;font-weight:bold;margin:5px 0">{value}</div>
        <div>{delta_html}</div>
    </div>
    """
