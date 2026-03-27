# Reddit WallStreetBets — Sentiment vs Stock Price Analysis

A data analysis project that explores whether Reddit sentiment from r/WallStreetBets predicts stock price movements, focusing on the January 2021 meme stock era.

## Live Dashboard

[wsb-sentiment-analysis-mlncreffkqsvllgrjygnrq.streamlit.app](https://wsb-sentiment-analysis-mlncreffkqsvllgrjygnrq.streamlit.app/)

---

## Project Overview

This project investigates the relationship between social media sentiment and stock returns using 53,187 Reddit posts from r/WallStreetBets. The analysis focuses on GME, AMC, and TSLA during the January–June 2021 period — one of the most dramatic episodes of retail investor activity in market history.

**Research Question:** Does Reddit sentiment lead, lag, or have no relationship with meme stock price movements?

---

## Key Findings

- No statistically significant lag correlation found between Reddit sentiment and returns at any lag from -5 to +5 days (all p-values > 0.05)
- GME's January 2021 short squeeze was driven by momentum and short squeeze mechanics, not sentiment alone
- Reddit post volume spiked from ~249 to ~2,940 posts/day during the peak squeeze period
- Sentiment turned slightly negative even as prices peaked, suggesting fear and uncertainty among retail investors
- This aligns with academic research showing sentiment is not a reliable standalone predictor of meme stock returns

---

## Dataset

- **Source:** [Reddit WallStreetBets Posts — Kaggle](https://www.kaggle.com/datasets/thomaskonstantin/reddit-wallstreetbets-posts-sentiment-analysis)
- **Size:** 53,187 posts
- **Period:** January 2021 — June 2021
- **Columns:** title, score, id, url, comms_num, created, body, timestamp

---

## Tech Stack

| Purpose | Library |
|---|---|
| Data manipulation | pandas, numpy |
| Sentiment analysis | vaderSentiment |
| Stock price data | yfinance |
| Statistical analysis | scipy, statsmodels |
| Visualization | matplotlib, seaborn |
| Dashboard | Streamlit |

---

## Project Structure

```
sentiment_stock_analysis/
│
├── data/
│   ├── raw/                     ← original dataset
│   └── processed/               ← cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_price_data.ipynb
│   └── 05_analysis.ipynb
│
├── dashboard/
│   └── app.py                   ← Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## Pipeline

1. **Data ingestion** — loaded 53,187 WSB posts from Kaggle CSV
2. **Text preprocessing** — cleaned titles, removed URLs and special characters, extracted stock tickers using regex
3. **Sentiment scoring** — applied VADER sentiment analysis to post titles, aggregated to daily scores weighted by post volume
4. **Price data** — fetched OHLCV data for GME, AMC, TSLA using yfinance, engineered daily return features
5. **Lag correlation analysis** — merged sentiment and price series, ran Pearson correlation across lags -5 to +5 days, tested statistical significance
6. **Dashboard** — built interactive Streamlit app with dynamic stock selection, volume toggle, and adjustable lag range

---

## Dashboard Features

- Select between GME, AMC, and TSLA
- Toggle volume overlay on price chart
- Adjust lag range with a slider
- All charts update dynamically
- Lag correlation table with p-values

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Manya-kh10/wsb-sentiment-analysis.git
cd wsb-sentiment-analysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard/app.py
```

---

## Author

**Manya** — B.Tech CSE (AI & ML), VIT Bhopal University

[GitHub](https://github.com/Manya-kh10)
