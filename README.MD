# 📊 YouTube Video Comment Sentiment Analyzer

A Streamlit web app that fetches the top comments from a YouTube video and analyzes their sentiment using a Hugging Face transformer model.

---

## 🚀 Features

- 🔍 Fetches up to **500 top-level comments** from a YouTube video using the YouTube Data API.
- 🤖 Performs **sentiment analysis (Positive / Negative / Neutral)** using a Hugging Face transformer model.
- ✅ Filters out comments containing links (to reduce spam and noise).
- 🌟 Displays the **most liked valid comment** (excluding link-containing ones).
- 📈 Summarizes the **overall sentiment distribution** across comments (e.g., % positive, % negative, % neutral).

---

## 📦 Tech Stack

- 🐍 Python
- 📺 YouTube Data API v3
- 🤗 Transformers (Hugging Face `pipeline`)
- 📊 Pandas
- ⚙️ Streamlit

---

## 🖥️ Demo

> (https://youtube-video-comment-sentiment-analysis-7.streamlit.app/)

---

## 📷 Screenshot

> ![alt text](image-1.png)

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/himadribh07/YouTube-Comment-Sentiment-Analysis.git
cd YouTube-Comment-Sentiment-Analysis
