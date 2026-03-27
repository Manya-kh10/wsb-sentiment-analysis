import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

st.set_page_config(page_title="WSB Sentiment Analysis", layout="wide")

st.title("Reddit WallStreetBets — Sentiment vs Stock Price")
st.markdown("Explore how Reddit sentiment correlates with meme stock movements")

@st.cache_data

def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))  
    data_path = os.path.join(base_dir, 'data', 'processed')

    df = pd.read_csv(os.path.join(data_path, 'wsb_cleaned.csv'))
    gme_s = pd.read_csv(os.path.join(data_path, 'gme_sentiment.csv'))
    amc_s = pd.read_csv(os.path.join(data_path, 'amc_sentiment.csv'))
    tsla_s = pd.read_csv(os.path.join(data_path, 'tsla_sentiment.csv'))
    gme_p = pd.read_csv(os.path.join(data_path, 'GME_price.csv'), skiprows=2)
    amc_p = pd.read_csv(os.path.join(data_path, 'AMC_price.csv'), skiprows=2)
    tsla_p = pd.read_csv(os.path.join(data_path, 'TSLA_price.csv'), skiprows=2)

    for p in [gme_p, amc_p, tsla_p]:
        p.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'daily_return']

    return df, {
        'GME': (gme_s, gme_p),
        'AMC': (amc_s, amc_p),
        'TSLA': (tsla_s, tsla_p)
    }
df, stock_data = load_data()

# ── Sidebar ──
st.sidebar.title("Controls")
selected_stock = st.sidebar.selectbox("Select Stock", ['GME', 'AMC', 'TSLA'])
show_volume = st.sidebar.checkbox("Show Volume", value=False)
lag_range = st.sidebar.slider("Lag Range (days)", min_value=3, max_value=10, value=5)

# ── Merge selected stock data ──
sentiment_df, price_df = stock_data[selected_stock]
sentiment_df['date_only'] = pd.to_datetime(sentiment_df['date_only'])
price_df['Date'] = pd.to_datetime(price_df['Date'])

merged = pd.merge(
    sentiment_df,
    price_df[['Date', 'Close', 'daily_return', 'Volume']],
    left_on='date_only',
    right_on='Date',
    how='inner'
).dropna()

# ── Metrics ──
st.subheader(f"{selected_stock} Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total WSB Posts", f"{len(df):,}")
col2.metric(f"{selected_stock} Posts", f"{len(sentiment_df):,}")
col3.metric("Avg Sentiment", f"{merged['avg_sentiment'].mean():.3f}")
col4.metric("Peak Price", f"${merged['Close'].max():.2f}")

st.divider()

# ── Price vs Sentiment Chart ──
st.subheader(f"{selected_stock} — Price vs Reddit Sentiment")

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(merged['date_only'], merged['Close'], color='steelblue', linewidth=2, label='Price')
ax1.set_ylabel('Price ($)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.bar(merged['date_only'], merged['avg_sentiment'],
        color=['green' if x > 0 else 'red' for x in merged['avg_sentiment']],
        alpha=0.5, label='Sentiment')
ax2.set_ylabel('Sentiment Score', color='gray')
ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

if show_volume:
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.fill_between(merged['date_only'], merged['Volume'],
                     alpha=0.1, color='purple', label='Volume')
    ax3.set_ylabel('Volume', color='purple')

plt.title(f'{selected_stock} Price vs Reddit Sentiment (Jan–Jun 2021)')
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ── Lag Correlation ──
st.subheader(f"Lag Correlation — Does Sentiment Predict {selected_stock} Returns?")

results = []
for lag in range(-lag_range, lag_range + 1):
    sentiment_shifted = merged['avg_sentiment'].shift(lag)
    combined = pd.DataFrame({
        'sentiment': sentiment_shifted,
        'returns': merged['daily_return']
    }).dropna()
    if len(combined) > 5:
        corr, pval = stats.pearsonr(combined['sentiment'], combined['returns'])
        results.append({'lag': lag, 'correlation': corr, 'p_value': pval})

lag_df = pd.DataFrame(results)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['green' if p < 0.05 else 'steelblue' for p in lag_df['p_value']]
    ax.bar(lag_df['lag'], lag_df['correlation'], color=colors)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag (days) — negative means sentiment leads price')
    ax.set_ylabel('Correlation')
    ax.set_title(f'Sentiment vs {selected_stock} Returns: Lag Correlation')
    ax.set_xticks(range(-lag_range, lag_range + 1))
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Lag Table")
    st.dataframe(
        lag_df.style.format({
            'correlation': '{:.3f}',
            'p_value': '{:.3f}'
        }),
        height=300
    )

st.divider()

# ── Post Volume Over Time ──
st.subheader(f"{selected_stock} — Reddit Post Volume Over Time")
fig, ax = plt.subplots(figsize=(14, 3))
ax.fill_between(merged['date_only'], merged['post_count'],
                color='orange', alpha=0.6)
ax.set_ylabel('Posts per Day')
ax.set_title(f'Daily {selected_stock} Mention Volume on r/WallStreetBets')
plt.tight_layout()
st.pyplot(fig)

st.divider()
st.caption("Data: Reddit WallStreetBets Posts (Kaggle) | Period: Jan–Jun 2021")