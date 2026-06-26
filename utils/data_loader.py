import pandas as pd
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"

@st.cache_data
def load_data():
    """Load all datasets with caching for performance"""
    data = {
        'products': pd.read_csv(DATA_PATH / 'products.csv'),
        'outlets': pd.read_csv(DATA_PATH / 'outlets.csv'),
        'sales_reps': pd.read_csv(DATA_PATH / 'sales_reps.csv'),
        'sales': pd.read_csv(DATA_PATH / 'sales_transactions.csv', parse_dates=['order_date']),
        'schemes': pd.read_csv(DATA_PATH / 'promotion_schemes.csv', parse_dates=['start_date', 'end_date']),
        'inventory': pd.read_csv(DATA_PATH / 'inventory_daily.csv', parse_dates=['snapshot_date']),
        'receivables': pd.read_csv(DATA_PATH / 'receivables.csv', parse_dates=['invoice_date', 'due_date', 'paid_date']),
        'expenses': pd.read_csv(DATA_PATH / 'expenses_monthly.csv', parse_dates=['expense_date']),
        'budget': pd.read_csv(DATA_PATH / 'budget_vs_actual.csv', parse_dates=['period']),
    }
    return data

def get_date_range(df, date_col='order_date'):
    return df[date_col].min(), df[date_col].max()
