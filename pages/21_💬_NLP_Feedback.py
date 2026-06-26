import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
from utils.data_loader import load_data
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NLP Feedback", page_icon="💬", layout="wide")
st.title("💬 Customer Feedback Analytics")
st.markdown("### NLP-Powered Insights from Customer Reviews")

# ============================================
# GENERATE SYNTHETIC FEEDBACK (since we don't have real reviews)
# ============================================
def generate_feedback_data():
    """Generate synthetic customer feedback based on transactional patterns"""
    np.random.seed(42)
    
    feedback_templates = {
        'positive': [
            "Excellent product quality, my family loves it!",
            "Great taste and value for money",
            "Fast delivery from {channel}, very satisfied",
            "Best {category} in the market, will buy again",
            "Amazing packaging and freshness",
            "Highly recommend this brand",
            "Consistent quality, never disappointed",
            "Good promotional offers helped save money",
            "Premium quality at affordable price",
            "Outstanding customer service from the outlet"
        ],
        'neutral': [
            "Product is okay, nothing special",
            "Average quality, expected better",
            "Delivery was on time but packaging could be better",
            "Price is reasonable",
            "Available at my regular store",
            "Standard product, meets expectations"
        ],
        'negative': [
            "Poor quality, very disappointed",
            "Product arrived damaged",
            "Overpriced compared to competitors",
            "Worst experience with this brand",
            "Out of stock at multiple outlets",
            "Expired product delivered",
            "Worst packaging, product was leaking",
            "Customer service was unhelpful",
            "Quality has degraded over time",
            "Would not recommend"
        ]
    }
    
    outlets = data['outlets']
    products = data['products']
    
    feedback = []
    for i in range(500):
        outlet = outlets.sample(1).iloc[0]
        product = products.sample(1).iloc[0]
        
        # Sentiment depends on channel and order patterns
        if outlet['outlet_type'] == 'E-commerce':
            sentiment = np.random.choice(['positive','neutral','negative'], p=[0.4, 0.3, 0.3])
        elif outlet['outlet_type'] == 'Modern Trade':
            sentiment = np.random.choice(['positive','neutral','negative'], p=[0.5, 0.3, 0.2])
        else:
            sentiment = np.random.choice(['positive','neutral','negative'], p=[0.35, 0.35, 0.3])
        
        text = np.random.choice(feedback_templates[sentiment])
        text = text.format(channel=outlet['outlet_type'], category=product['category'])
        
        feedback.append({
            'feedback_id': f'FB{i:05d}',
            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'outlet_id': outlet['outlet_id'],
            'product_id': product['product_id'],
            'category': product['category'],
            'brand': product['brand'],
            'channel': outlet['outlet_type'],
            'region': outlet['region'],
            'rating': {'positive': np.random.choice([4, 5]), 
                       'neutral': 3, 
                       'negative': np.random.choice([1, 2])}[sentiment],
            'feedback_text': text,
            'true_sentiment': sentiment
        })
    
    return pd.DataFrame(feedback)

# ============================================
# SIMPLE RULE-BASED SENTIMENT (or use VADER if available)
# ============================================
def analyze_sentiment(text):
    """Simple lexicon-based sentiment analyzer"""
    positive_words = ['excellent','great','amazing','best','love','good','satisfied','recommend',
                      'outstanding','premium','consistent','affordable','helpful','fast']
    negative_words = ['poor','worst','disappointed','damaged','overpriced','bad','expired',
                      'leaking','unhelpful','degraded','late','broken','terrible','awful']
    
    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        return 'positive', (pos_count - neg_count) / (pos_count + neg_count + 1)
    elif neg_count > pos_count:
        return 'negative', (neg_count - pos_count) / (pos_count + neg_count + 1)
    else:
        return 'neutral', 0.5

# Try VADER for better accuracy
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment_vader(text):
        scores = analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 'positive', scores['compound']
        elif scores['compound'] <= -0.05:
            return 'negative', abs(scores['compound'])
        else:
            return 'neutral', 0.5
    USE_VADER = True
except ImportError:
    USE_VADER = False

# ============================================
# LOAD OR GENERATE FEEDBACK
# ============================================
data = load_data()

if 'feedback_df' not in st.session_state:
    with st.spinner("Loading customer feedback..."):
        st.session_state['feedback_df'] = generate_feedback_data()

feedback_df = st.session_state['feedback_df']

# Analyze sentiment
if USE_VADER:
    feedback_df[['sentiment', 'sentiment_score']] = feedback_df['feedback_text'].apply(
        lambda x: pd.Series(analyze_sentiment_vader(x))
    )
else:
    feedback_df[['sentiment', 'sentiment_score']] = feedback_df['feedback_text'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

# ============================================
# KPIs
# ============================================
st.subheader("📊 Feedback Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Feedback", len(feedback_df))
c2.metric("Avg Rating", f"⭐ {feedback_df['rating'].mean():.2f}")
c3.metric("Positive %", f"{(feedback_df['sentiment']=='positive').mean()*100:.1f}%")
c4.metric("NPS Score", f"{((feedback_df['rating']>=4).mean() - (feedback_df['rating']<=2).mean())*100:.0f}")

# ============================================
# SENTIMENT DISTRIBUTION
# ============================================
c1, c2 = st.columns(2)

with c1:
    sentiment_counts = feedback_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.pie(sentiment_counts, names='Sentiment', values='Count', 
                 title='Overall Sentiment Distribution',
                 color='Sentiment',
                 color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    channel_sentiment = feedback_df.groupby(['channel', 'sentiment']).size().reset_index(name='count')
    fig = px.bar(channel_sentiment, x='channel', y='count', color='sentiment',
                 title='Sentiment by Channel',
                 color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'},
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# KEY TOPICS / KEYWORDS
# ============================================
st.subheader("🔑 Top Keywords Analysis")

from collections import Counter
import re

def extract_keywords(texts, top_n=20):
    stop_words = {'the','a','an','is','was','are','were','be','been','being','have','has',
                  'had','do','does','did','will','would','could','should','may','might','must',
                  'shall','can','need','dare','ought','used','to','of','in','for','on','with',
                  'at','by','from','as','into','through','during','before','after','above','below',
                  'between','out','against','this','that','these','those','i','me','my','we','our'}
    
    words = []
    for text in texts:
        words.extend([w.lower() for w in re.findall(r'\b[a-z]{4,}\b', text) 
                     if w.lower() not in stop_words])
    
    return Counter(words).most_common(top_n)

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**😊 Positive Keywords**")
    pos_texts = feedback_df[feedback_df['sentiment']=='positive']['feedback_text']
    pos_kw = extract_keywords(pos_texts, 15)
    pos_df = pd.DataFrame(pos_kw, columns=['Keyword', 'Count'])
    fig = px.bar(pos_df.head(10), x='Count', y='Keyword', orientation='h',
                 color='Count', color_continuous_scale='Greens')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("**😐 Neutral Keywords**")
    neu_texts = feedback_df[feedback_df['sentiment']=='neutral']['feedback_text']
    neu_kw = extract_keywords(neu_texts, 15)
    if neu_kw:
        neu_df = pd.DataFrame(neu_kw, columns=['Keyword', 'Count'])
        fig = px.bar(neu_df.head(10), x='Count', y='Keyword', orientation='h',
                     color='Count', color_continuous_scale='Greys')
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.write("**😞 Negative Keywords**")
    neg_texts = feedback_df[feedback_df['sentiment']=='negative']['feedback_text']
    neg_kw = extract_keywords(neg_texts, 15)
    neg_df = pd.DataFrame(neg_kw, columns=['Keyword', 'Count'])
    fig = px.bar(neg_df.head(10), x='Count', y='Keyword', orientation='h',
                 color='Count', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# BRAND & CATEGORY SENTIMENT
# ============================================
st.subheader("📦 Sentiment by Brand & Category")

brand_sentiment = feedback_df.groupby(['brand', 'sentiment']).size().reset_index(name='count')
brand_pivot = brand_sentiment.pivot(index='brand', columns='sentiment', values='count').fillna(0)
brand_pivot['total'] = brand_pivot.sum(axis=1)
brand_pivot['positive_pct'] = (brand_pivot.get('positive', 0) / brand_pivot['total'] * 100).round(1)
brand_pivot = brand_pivot.sort_values('positive_pct', ascending=False)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(brand_pivot.reset_index(), x='brand', y='positive_pct',
                 color='positive_pct', color_continuous_scale='RdYlGn',
                 title='Brand Sentiment Score')
    st.plotly_chart(fig, use_container_width=True)

with c2:
    cat_sentiment = feedback_df.groupby('category')['rating'].agg(['mean','count']).reset_index()
    cat_sentiment.columns = ['Category', 'Avg Rating', 'Feedback Count']
    fig = px.scatter(cat_sentiment, x='Feedback Count', y='Avg Rating', 
                     size='Feedback Count', color='Avg Rating', hover_data=['Category'],
                     color_continuous_scale='RdYlGn', title='Category Performance',
                     size_max=50)
    fig.add_hline(y=3.5, line_dash="dash", line_color="red", annotation_text="Threshold")
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# FEEDBACK EXPLORER
# ============================================
st.markdown("---")
st.subheader("🔍 Feedback Explorer")

filter_sentiment = st.multiselect("Filter by Sentiment", 
                                   ['positive','neutral','negative'],
                                   default=['negative'])

filtered_fb = feedback_df[feedback_df['sentiment'].isin(filter_sentiment)]

st.dataframe(filtered_fb[['date','channel','region','brand','category','rating','sentiment','feedback_text']].head(50),
             use_container_width=True)

# ============================================
# ACTIONABLE INSIGHTS
# ============================================
st.markdown("---")
st.subheader("💡 Actionable Insights")

# Find problematic areas
worst_brand = brand_pivot.iloc[-1].name
worst_cat = cat_sentiment.sort_values('Avg Rating').iloc[0]['Category']
nps_score = ((feedback_df['rating']>=4).mean() - (feedback_df['rating']<=2).mean()) * 100

col1, col2 = st.columns(2)
with col1:
    if (feedback_df['sentiment']=='negative').mean() > 0.25:
        st.error(f"""
        🚨 **High Negative Sentiment**
        - {(feedback_df['sentiment']=='negative').mean()*100:.1f}% of feedback is negative
        - **Action**: Investigate {worst_brand} brand quality issues
        """)
    else:
        st.success("✅ Sentiment is within acceptable range")

with col2:
    if nps_score < 30:
        st.warning(f"""
        ⚠️ **Low NPS Score: {nps_score:.0f}**
        - Below industry benchmark (30+)
        - Focus on {worst_cat} category improvement
        """)
    else:
        st.success(f"✅ Strong NPS Score: {nps_score:.0f}")
