import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
import plotly.io as pio
from utils.data_loader import load_data
from utils.kpi_calculator import calculate_kpis

st.set_page_config(page_title="Export Reports", page_icon="📤", layout="wide")
st.title("📤 Export Center - PDF & Excel Reports")

data = load_data()

tab1, tab2 = st.tabs(["📄 PDF Reports", "📊 Excel Exports"])

# ============================================
# PDF REPORTS
# ============================================
with tab1:
    st.subheader("📄 Generate CFO PDF Report")
    
    report_type = st.selectbox("Report Type", [
        "📊 Executive Summary",
        "💰 Profitability Report",
        "💵 Working Capital Report",
        "📦 Inventory Report",
        "🎯 Promotion ROI Report"
    ])
    
    period_start = st.date_input("From", data['sales']['order_date'].min())
    period_end = st.date_input("To", data['sales']['order_date'].max())
    
    def generate_pdf(report_type, data, period_start, period_end):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                       textColor=colors.HexColor('#667eea'),
                                       alignment=1, fontSize=24)
        elements = []
        
        # Title
        elements.append(Paragraph("CFO Intelligent Platform", title_style))
        elements.append(Paragraph(f"{report_type}", styles['Heading2']))
        elements.append(Paragraph(f"Period: {period_start} to {period_end}", styles['Normal']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        sales = data['sales']
        sales_f = sales[(sales['order_date'].dt.date >= period_start) & 
                        (sales['order_date'].dt.date <= period_end)]
        
        # KPIs
        kpis = calculate_kpis(sales_f)
        kpi_data = [[k, v] for k, v in kpis.items()]
        kpi_table = Table(kpi_data, colWidths=[3*inch, 3*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#dee2e6')),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold')
        ]))
        elements.append(Paragraph("Executive Summary", styles['Heading2']))
        elements.append(kpi_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Channel breakdown
        elements.append(Paragraph("Revenue by Channel", styles['Heading2']))
        channel_df = sales_f.groupby('channel')['net_amount'].sum().reset_index()
        channel_data = [channel_df.columns.tolist()] + channel_df.values.tolist()
        ch_table = Table(channel_data)
        ch_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
        ]))
        elements.append(ch_table)
        
        # Top products
        elements.append(Spacer(1, 0.3*inch))
        elements.append(PageBreak())
        elements.append(Paragraph("Top 10 Products", styles['Heading2']))
        top = sales_f.groupby('product_id')['net_amount'].sum().nlargest(10).reset_index()
        top = top.merge(data['products'][['product_id','product_name']], on='product_id')
        top_data = [['Product', 'Revenue']] + top[['product_name','net_amount']].values.tolist()
        top_table = Table(top_data, colWidths=[4*inch, 2*inch])
        top_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#28a745')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ]))
        elements.append(top_table)
        
        doc.build(elements)
        buffer.seek(0)
        return buffer
    
    if st.button("🔄 Generate PDF Report"):
        pdf_buffer = generate_pdf(report_type, data, period_start, period_end)
        st.success("✅ Report generated!")
        st.download_button(
            label="📥 Download PDF",
            data=pdf_buffer,
            file_name=f"CFO_Report_{report_type}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

# ============================================
# EXCEL EXPORTS
# ============================================
with tab2:
    st.subheader("📊 Excel Multi-Sheet Export")
    
    export_type = st.selectbox("What to Export", [
        "Complete Sales Data",
        "Sales + Products + Outlets (Joined)",
        "Receivables Aging",
        "Inventory Snapshot",
        "Budget vs Actual"
    ])
    
    def to_excel(df_dict):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for sheet_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        buffer.seek(0)
        return buffer
    
    if st.button("🔄 Generate Excel"):
        if export_type == "Complete Sales Data":
            df_dict = {'Sales': data['sales'], 'Products': data['products'], 'Outlets': data['outlets']}
        elif export_type == "Sales + Products + Outlets (Joined)":
            joined = data['sales'].merge(data['products'], on='product_id', suffixes=('_s','_p'))
            joined = joined.merge(data['outlets'], on='outlet_id', suffixes=('','_o'))
            df_dict = {'Master Data': joined}
        elif export_type == "Receivables Aging":
            df_dict = {'Receivables': data['receivables'], 'Outlets': data['outlets']}
        elif export_type == "Inventory Snapshot":
            df_dict = {'Inventory': data['inventory'], 'Products': data['products']}
        else:
            df_dict = {'Budget': data['budget']}
        
        excel_buffer = to_excel(df_dict)
        st.success("✅ Excel ready!")
        st.download_button(
            label="📥 Download Excel",
            data=excel_buffer,
            file_name=f"CFO_Data_{export_type.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
