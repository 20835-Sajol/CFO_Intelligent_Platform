import pandas as pd
import numpy as np

def calculate_kpis(sales_df):
    """Top-level CFO KPIs"""
    total_revenue = sales_df['net_amount'].sum()
    total_cogs = sales_df['cogs'].sum()
    gross_profit = total_revenue - total_cogs
    gross_margin_pct = (gross_profit / total_revenue) * 100
    
    return {
        'Total Revenue': f"₹{total_revenue/1e7:.2f} Cr",
        'Gross Profit': f"₹{gross_profit/1e7:.2f} Cr",
        'Gross Margin %': f"{gross_margin_pct:.2f}%",
        'Total Orders': f"{sales_df['order_id'].nunique():,}",
        'Active SKUs': f"{sales_df['product_id'].nunique():,}",
        'Active Outlets': f"{sales_df['outlet_id'].nunique():,}"
    }

def working_capital_metrics(receivables_df):
    """DSO, DPO, DIO calculation"""
    avg_receivable = receivables_df[receivables_df['status'] != 'Paid']['invoice_amount'].mean()
    total_outstanding = receivables_df[receivables_df['status'] != 'Paid']['outstanding'].sum()
    overdue = receivables_df[receivables_df['status'] == 'Overdue']['outstanding'].sum()
    collection_rate = (receivables_df['paid_amount'].sum() / receivables_df['invoice_amount'].sum()) * 100
    
    return {
        'Total Outstanding': f"₹{total_outstanding/1e7:.2f} Cr",
        'Overdue Amount': f"₹{overdue/1e7:.2f} Cr",
        'Collection Rate': f"{collection_rate:.1f}%",
        'Avg Outstanding/Invoice': f"₹{avg_receivable:,.0f}"
    }

def revenue_by_period(sales_df, freq='M'):
    return sales_df.set_index('order_date').resample(freq)['net_amount'].sum().reset_index()

def revenue_by_dimension(sales_df, dim):
    return sales_df.groupby(dim)['net_amount'].sum().reset_index().sort_values('net_amount', ascending=False)

def pareto_analysis(sales_df, dim='product_id'):
    """80/20 Pareto analysis"""
    grouped = sales_df.groupby(dim)['net_amount'].sum().sort_values(ascending=False).reset_index()
    grouped['cum_pct'] = grouped['net_amount'].cumsum() / grouped['net_amount'].sum() * 100
    return grouped
