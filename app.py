import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px

st.set_page_config(page_title="AI Social Media Dashboard", layout="wide")
st.title("📊 AI-Powered Social Media Dashboard")

df = pd.read_csv("data.csv")
df['Date'] = pd.to_datetime(df['Date'])

platform = st.selectbox("Select Platform", df['Platform'].unique())
df_p = df[df['Platform'] == platform]

# 1. Show current data
st.subheader(f"{platform} Growth")
fig = px.line(df_p, x='Date', y='Followers', title='Historical Followers', markers=True)
st.plotly_chart(fig, use_container_width=True)

# 2. ML Prediction
st.subheader("🤖 30-Day Forecast")
X = np.arange(len(df_p)).reshape(-1, 1)  # Days as numbers
y = df_p['Followers'].values

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(df_p), len(df_p) + 30).reshape(-1, 1)
predictions = model.predict(future_days)

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Followers in 30 days", f"{int(predictions[-1]):,}")
with col2:
    growth = int(predictions[-1] - df_p['Followers'].iloc[-1])
    st.metric("Expected Growth", f"+{growth:,}")
    from textblob import TextBlob

st.subheader("💬 AI Sentiment Analysis")

# Load comments
try:
    comments_df = pd.read_csv("comments.csv")
    comments_df_p = comments_df[comments_df['Platform'] == platform]
    
    # Analyze sentiment
    def get_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    
    comments_df_p['Sentiment'] = comments_df_p['Comment'].apply(get_sentiment)
    
    pos = len(comments_df_p[comments_df_p['Sentiment'] == 'Positive'])
    neg = len(comments_df_p[comments_df_p['Sentiment'] == 'Negative'])
    total = len(comments_df_p)
    
    if total > 0:
        st.metric("Positive Comments", f"{int(pos/total*100)}%")
        st.dataframe(comments_df_p[['Comment', 'Sentiment']])
    else:
        st.info("No comments for this platform yet")
        
except:
    st.warning("Create comments.csv file to enable sentiment analysis")