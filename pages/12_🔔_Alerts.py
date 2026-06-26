import streamlit as st
import pandas as pd
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
from utils.data_loader import load_data

st.set_page_config(page_title="Alerts & Notifications", page_icon="🔔", layout="wide")
st.title("🔔 Real-Time Alerts & Notifications")

data = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Alert Rules", "📊 Active Alerts", "📧 Email Config", "💬 Slack/Teams"])

# ============================================
# ALERT RULES
# ============================================
with tab1:
    st.subheader("⚙️ Configure Alert Rules")
    
    if 'alert_rules' not in st.session_state:
        st.session_state.alert_rules = []
    
    col1, col2 = st.columns(2)
    with col1:
        alert_type = st.selectbox("Alert Type", [
            "📉 Revenue Drop", "📈 Revenue Spike", "📦 Low Stock",
            "💰 Overdue Invoice", "🚨 Margin Below Threshold",
            "📊 Budget Variance", "🔍 Unusual Transaction"
        ])
        threshold = st.number_input("Threshold Value", value=10.0)
        comparison = st.selectbox("Comparison", ["Above", "Below"])
    
    with col2:
        dimension = st.selectbox("Dimension", ["Overall", "Region", "Channel", "Category", "Brand", "Outlet"])
        notification_method = st.multiselect("Notify via", ["Email", "Slack", "Teams", "Dashboard"])
        frequency = st.selectbox("Frequency", ["Real-time", "Hourly", "Daily", "Weekly"])
    
    if st.button("➕ Add Alert Rule"):
        rule = {
            'id': len(st.session_state.alert_rules) + 1,
            'type': alert_type,
            'threshold': threshold,
            'comparison': comparison,
            'dimension': dimension,
            'notify': notification_method,
            'frequency': frequency,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'status': 'Active'
        }
        st.session_state.alert_rules.append(rule)
        st.success(f"✅ Alert rule #{rule['id']} added!")
    
    if st.session_state.alert_rules:
        st.subheader("📋 Configured Rules")
        rules_df = pd.DataFrame(st.session_state.alert_rules)
        st.dataframe(rules_df, use_container_width=True)
        
        # Save/Load
        rules_json = json.dumps(st.session_state.alert_rules, indent=2)
        st.download_button("💾 Export Rules", rules_json, "alert_rules.json", "application/json")

# ============================================
# ACTIVE ALERTS (Live Detection)
# ============================================
with tab2:
    st.subheader("📊 Active Alerts - Live Detection")
    
    # Auto-detect current alerts
    alerts_detected = []
    
    # 1. Revenue drop detection
    sales = data['sales'].copy()
    sales['month'] = sales['order_date'].dt.to_period('M')
    monthly = sales.groupby('month')['net_amount'].sum().reset_index()
    monthly['mom_growth'] = monthly['net_amount'].pct_change() * 100
    
    for _, row in monthly.iterrows():
        if pd.notna(row['mom_growth']) and row['mom_growth'] < -10:
            alerts_detected.append({
                'severity': '🔴 HIGH',
                'type': '📉 Revenue Drop',
                'message': f"Revenue dropped {abs(row['mom_growth']):.1f}% in {row['month']}",
                'value': f"₹{row['net_amount']/1e7:.2f} Cr",
                'time': row['month'].strftime('%Y-%m')
            })
    
    # 2. Overdue receivables
    ar = data['receivables']
    overdue = ar[ar['status'] == 'Overdue']
    if len(overdue) > 0:
        alerts_detected.append({
            'severity': '🟡 MEDIUM',
            'type': '💰 Overdue Invoices',
            'message': f"{len(overdue)} invoices overdue worth ₹{overdue['outstanding'].sum()/1e5:.1f} L",
            'value': f"{len(overdue)} invoices",
            'time': 'Now'
        })
    
    # 3. Bad debt
    bad_debt = ar[ar['status'] == 'Bad Debt']
    if len(bad_debt) > 0:
        alerts_detected.append({
            'severity': '🔴 HIGH',
            'type': '🚨 Bad Debt',
            'message': f"₹{bad_debt['outstanding'].sum()/1e5:.1f} L marked as bad debt",
            'value': f"{len(bad_debt)} invoices",
            'time': 'Now'
        })
    
    # 4. Low stock
    inv = data['inventory']
    last_date = inv['snapshot_date'].max()
    recent = inv[inv['snapshot_date'] == last_date]
    low_stock = recent[recent['closing_stock'] < recent['opening_stock'] * 0.3]
    if len(low_stock) > 0:
        alerts_detected.append({
            'severity': '🟡 MEDIUM',
            'type': '📦 Low Stock Alert',
            'message': f"{len(low_stock)} SKUs below 30% of opening stock",
            'value': f"{len(low_stock)} SKUs",
            'time': last_date.strftime('%Y-%m-%d')
        })
    
    # 5. Budget variance
    budget = data['budget']
    over_budget = budget[budget['variance_pct'] > 15]
    if len(over_budget) > 0:
        alerts_detected.append({
            'severity': '🟡 MEDIUM',
            'type': '📊 Budget Overshoot',
            'message': f"{len(over_budget)} region/channel combinations over budget by 15%+",
            'value': f"{len(over_budget)} entries",
            'time': 'Current'
        })
    
    # Display
    if alerts_detected:
        st.error(f"🚨 {len(alerts_detected)} active alerts detected!")
        alerts_df = pd.DataFrame(alerts_detected)
        st.dataframe(alerts_df, use_container_width=True)
        
        # Severity breakdown
        sev_count = alerts_df['severity'].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 High Severity", len(alerts_df[alerts_df['severity'].str.contains('HIGH')]))
        c2.metric("🟡 Medium Severity", len(alerts_df[alerts_df['severity'].str.contains('MEDIUM')]))
        c3.metric("Total Alerts", len(alerts_df))
    else:
        st.success("✅ No critical alerts at this time")

# ============================================
# EMAIL CONFIG
# ============================================
with tab3:
    st.subheader("📧 Email Notification Setup")
    
    with st.expander("Configure SMTP"):
        smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
        smtp_port = st.number_input("Port", value=587)
        sender_email = st.text_input("Sender Email")
        sender_password = st.text_input("App Password", type="password")
        recipient_emails = st.text_area("Recipients (comma-separated)")
    
    if st.button("🧪 Send Test Email"):
        if sender_email and sender_password:
            try:
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = recipient_emails
                msg['Subject'] = f"🚨 CFO Platform Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
                body = f"""
                <h2>CFO Platform Alert Summary</h2>
                <p><strong>Time:</strong> {datetime.now()}</p>
                <p><strong>Active Alerts:</strong> {len(alerts_detected)}</p>
                <ul>
                {"".join(f"<li>{a['type']}: {a['message']}</li>" for a in alerts_detected[:10])}
                </ul>
                """
                msg.attach(MIMEText(body, 'html'))
                
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.send_message(msg)
                st.success("✅ Test email sent successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Configure SMTP credentials first")

# ============================================
# SLACK / TEAMS
# ============================================
with tab4:
    st.subheader("💬 Slack / Microsoft Teams Webhook")
    
    platform = st.radio("Platform", ["Slack", "Microsoft Teams"])
    webhook_url = st.text_input(f"{platform} Webhook URL", type="password")
    
    if st.button("🧪 Send Test Notification"):
        if webhook_url:
            try:
                if platform == "Slack":
                    payload = {
                        "text": f"🚨 *CFO Platform Alert*\nActive Alerts: {len(alerts_detected)}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    }
                else:  # Teams
                    payload = {
                        "@type": "MessageCard",
                        "summary": "CFO Alert",
                        "title": f"🚨 CFO Platform Alert",
                        "text": f"Active Alerts: {len(alerts_detected)}\nTime: {datetime.now()}"
                    }
                
                response = requests.post(webhook_url, json=payload)
                if response.status_code == 200:
                    st.success(f"✅ {platform} notification sent!")
                else:
                    st.error(f"❌ Failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Enter webhook URL")

# Sidebar - Alert summary widget
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔔 Alert Summary")
    if alerts_detected:
        st.error(f"🚨 {len(alerts_detected)} Active Alerts")
        for a in alerts_detected[:3]:
            st.write(f"{a['severity']} {a['type']}")
    else:
        st.success("✅ All Clear")
