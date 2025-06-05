

# # In[2]:


# import re
# import streamlit as st
# import pandas as pd
# from googleapiclient.discovery import build
# from transformers import pipeline

# # --- Set up API Key ---
# api_key = st.secrets["api_key"]  # or paste directly for local testing

# # --- Sentiment Pipeline ---

# sentiment_pipeline = pipeline(
#     "sentiment-analysis"
# )


# # --- Helper to extract video ID ---
# def extract_video_id(url):
#     patterns = [
#         r'youtu\.be/([^?&]+)',
#         r'youtube\.com/watch\?v=([^?&]+)',
#         r'youtube\.com/embed/([^?&]+)'
#     ]
#     for pattern in patterns:
#         match = re.search(pattern, url)
#         if match:
#             return match.group(1)
#     return None

# # --- Fetch Comments ---
# def get_comments_with_likes_filtered(video_id, max_comments=500):
#     youtube = build('youtube', 'v3', developerKey=api_key)
#     comments = []
#     next_page_token = None

#     while len(comments) < max_comments:
#         request = youtube.commentThreads().list(
#             part="snippet",
#             videoId=video_id,
#             maxResults=100,
#             pageToken=next_page_token,
#             textFormat="plainText"
#         )
#         response = request.execute()

#         for item in response['items']:
#             snippet = item['snippet']['topLevelComment']['snippet']
#             text = snippet['textDisplay']
#             like_count = snippet.get('likeCount', 0)

#             if "http" not in text and "www" not in text:
#                 comments.append({
#                     'text': text,
#                     'likes': like_count
#                 })

#             if len(comments) >= max_comments:
#                 break

#         next_page_token = response.get("nextPageToken")
#         if not next_page_token:
#             break

#     return comments

# # --- Analyze Sentiment ---
# def analyze_sentiment(comments):
#     results = []
#     for comment in comments:
#         if comment.strip():
#             result = sentiment_pipeline(comment[:512])[0]
#             results.append({
#                 "comment": comment,
#                 "label": result["label"],
#                 "score": round(result["score"], 3)
#             })
#     return pd.DataFrame(results)

# # --- Streamlit UI ---
# st.title("📊 YouTube Video Comment Sentiment Analysis")
# video_url = st.text_input("Enter YouTube Video URL:")

# if st.button("Analyze"):
#     with st.spinner("Fetching and analyzing comments..."):
#         video_id = extract_video_id(video_url)
#         if not video_id:
#             st.error("❌ Invalid YouTube URL.")
#         else:
#             comments = get_comments_with_likes_filtered(video_id)
#             comment_texts = [c['text'] for c in comments]
#             df = analyze_sentiment(comment_texts)

#             st.success("✅ Analysis Complete!")

#             # Show sentiment summary
#             sentiment_counts = df['label'].value_counts()
#             pos = sentiment_counts.get("POSITIVE", 0)
#             neg = sentiment_counts.get("NEGATIVE", 0)
#             total = pos + neg
#             pos_percent = round((pos / total) * 100, 1) if total > 0 else 0
#             neg_percent = round((neg / total) * 100, 1) if total > 0 else 0

#             st.subheader("🔍 Sentiment Summary")
#             st.write(f"👍 Positive: {pos} ({pos_percent}%)")
#             st.write(f"👎 Negative: {neg} ({neg_percent}%)")

#             if pos_percent >= 70:
#                 verdict = "✅ People really like this video!"
#             elif pos_percent >= 50:
#                 verdict = "🙂 Mostly positive."
#             else:
#                 verdict = "⚠️ Mostly negative."
#             st.write("🎯 Verdict:", verdict)

#             # Show most liked comment
#             most_liked = max(comments, key=lambda c: c['likes'], default=None)
#             if most_liked:
#                 st.subheader("⭐ Most Liked Comment")
#                 st.write(f"💬 {most_liked['text']}")
#                 st.write(f"👍 Likes: {most_liked['likes']}")

#             # Show all results
#             st.subheader("📋 All Comments & Sentiments")
#             st.dataframe(df)


# # In[ ]:


import re
import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline
import torch
torch.classes.__path__ = [] # Add this line

# ... rest of your code
# --- Set up API Key with error handling ---
try:
    api_key = st.secrets["api_key"]
except Exception as e:
    st.error(f"Error loading API key: {str(e)}")
    st.stop()

# --- Sentiment Pipeline ---
@st.cache_resource  #  CRITICAL:  Cache the pipeline using @st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )
    except Exception as e:
        st.error(f"Error loading sentiment model: {str(e)}")
        st.stop()

sentiment_pipeline = load_sentiment_pipeline()  # Load the pipeline once

# --- Helper to extract video ID ---
def extract_video_id(url):
    patterns = [
        r'youtu\.be/([^?&]+)',
        r'youtube\.com/watch\?v=([^?&]+)',
        r'youtube\.com/embed/([^?&]+)',
        r'youtube\.com/shorts/([^?&]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# --- Fetch Comments with better error handling ---
import time
import random

def get_comments_with_likes_filtered(video_id, max_comments=500, max_retries=3):
    for attempt in range(max_retries):
        try:
            # ... your existing code ...
            return comments
        except HttpError as e:
            if e.resp.status in [429, 500, 503]:  # Retryable errors
                wait_time = (2 ** attempt) + random.random()  # Exponential backoff with jitter
                st.warning(f"YouTube API error: {str(e)}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"YouTube API error: {str(e)}")
                return []
        except Exception as e:
            st.error(f"Error fetching comments: {str(e)}")
            return []
    st.error("Maximum retries exceeded for YouTube API.")
    return []

# Apply similar retry logic to load_sentiment_pipeline() if you suspect network issues there

# --- Analyze Sentiment ---
def analyze_sentiment(comments, batch_size=32):  # Add batch_size
    results = []
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        try:
            batch_results = sentiment_pipeline(batch)  # Process a batch
            for comment, result in zip(batch, batch_results):
                results.append({
                    "comment": comment,
                    "label": result["label"],
                    "score": round(result["score"], 3)
                })
        except Exception as e:
            st.warning(f"Couldn't analyze a batch of comments: {str(e)}")
    return pd.DataFrame(results)

# ... in your main block:
df = analyze_sentiment(comment_texts)  #  Use the function

# --- Streamlit UI ---
st.title("📊 YouTube Video Comment Sentiment Analysis")
st.write("Analyze the sentiment of comments on any YouTube video")

video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Comments"):
    if not video_url:
        st.warning("Please enter a YouTube URL")
    else:
        with st.spinner("Fetching and analyzing comments..."):
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL. Please check the URL and try again.")
            else:
                comments = get_comments_with_likes_filtered(video_id)
                
                if not comments:
                    st.error("No comments found or couldn't fetch comments.")
                else:
                    comment_texts = [c['text'] for c in comments]
                    df = analyze_sentiment(comment_texts)

                    if df.empty:
                        st.warning("No comments could be analyzed.")
                    else:
                        st.success("✅ Analysis Complete!")

                        # Sentiment summary
                        st.subheader("🔍 Sentiment Summary")
                        sentiment_counts = df['label'].value_counts()
                        pos = sentiment_counts.get("POSITIVE", 0)
                        neg = sentiment_counts.get("NEGATIVE", 0)
                        total = pos + neg

                        if total > 0:
                            pos_percent = round((pos / total) * 100, 1)
                            neg_percent = round((neg / total) * 100, 1)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Positive Comments", f"{pos} ({pos_percent}%)")
                            with col2:
                                st.metric("Negative Comments", f"{neg} ({neg_percent}%)")
                            
                            # Verdict
                            if pos_percent >= 70:
                                verdict = "✅ People really like this video!"
                            elif pos_percent >= 50:
                                verdict = "🙂 Mostly positive."
                            else:
                                verdict = "⚠️ Mostly negative."
                            st.info(verdict)

                            # Most liked comment
                            most_liked = max(comments, key=lambda c: c['likes'])
                            with st.expander("⭐ Most Liked Comment"):
                                st.write(most_liked['text'])
                                st.caption(f"👍 {most_liked['likes']} likes")

                            # Show all results
                            with st.expander("📋 View All Comments & Sentiments"):
                                st.dataframe(df)

