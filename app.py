import re
import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline
import torch
import time
import random

# --- Streamlit Page Configuration ---
# THIS MUST BE THE FIRST STREAMLIT COMMAND IN YOUR SCRIPT
st.set_page_config(page_title="YouTube Comment Sentiment", page_icon="📊", layout="centered") # Changed layout to "centered"

# --- PyTorch/Streamlit Compatibility Fix ---
torch.classes.__path__ = []

# --- Set up API Key with error handling ---
try:
    api_key = st.secrets["api_key"]
except Exception as e:
    st.error(f"Error loading API key: {str(e)}")
    st.stop() # Stop the app if API key is not found

# --- Sentiment Pipeline ---
@st.cache_resource
def load_sentiment_pipeline():
    """Loads the pre-trained sentiment analysis pipeline."""
    try:
        # Using a smaller, faster model suitable for general sentiment analysis.
        return pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )
    except Exception as e:
        st.error(f"Error loading sentiment model: {str(e)}")
        st.stop() # Stop the app if the model fails to load

sentiment_pipeline = load_sentiment_pipeline() # Initialize the pipeline when the script runs

# --- Helper to extract video ID ---
def extract_video_id(url):
    """
    Extracts the YouTube video ID from various YouTube URL formats.
    """
    patterns = [
        r'youtu\.be/([^?&]+)',              # Shortened URL (youtu.be)
        r'youtube\.com/watch\?v=([^?&]+)',  # Standard watch URL
        r'youtube\.com/embed/([^?&]+)',     # Embed URL
        r'youtube\.com/shorts/([^?&]+)'     # Shorts URL
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# --- Fetch Comments with Retry Logic ---
def get_comments_with_likes_filtered(video_id, max_comments=500, max_retries=5):
    """
    Fetches comments for a given YouTube video ID, including like counts,
    with retry logic for API errors and basic spam filtering.
    """
    comments = [] # Initialize comments list
    next_page_token = None
    
    # st.info(f"Attempting to fetch comments for video ID: {video_id}") # COMMENTED OUT
    st.write("Fetching comments...") # Keep a single general message if you want
    
    for attempt in range(max_retries):
        try:
            youtube = build('youtube', 'v3', developerKey=api_key)

            while len(comments) < max_comments:
                # st.write(f"Fetching page {len(comments) // 100 + 1} (Attempt {attempt + 1})...") # COMMENTED OUT
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100, # Max results per page allowed by YouTube API
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()
                
                # --- Debugging: Inspect raw API response ---
                # st.json(response) # Uncomment temporarily to see the full API response
                # if not response.get('items'):
                #     st.warning("YouTube API returned no items for this page.")


                for item in response.get('items', []):
                    snippet = item['snippet']['topLevelComment']['snippet']
                    text = snippet['textDisplay']
                    like_count = snippet.get('likeCount', 0)

                    # Basic filtering for spam/irrelevant comments
                    # --- Debugging: Temporarily disable filtering if you suspect it ---
                    # if True: # Use this to disable filtering and see if comments are fetched
                    if len(text) > 5 and "http" not in text and "www" not in text:
                        comments.append({
                            'text': text,
                            'likes': like_count
                        })

                if len(comments) >= max_comments:
                    st.info(f"Collected {len(comments)} comments. Stopping.")
                    break # Stop if we've collected enough comments

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    st.info("No more comment pages found.")
                    break # No more pages of comments

            return comments # Successfully fetched comments

        except HttpError as e:
            st.error(f"HTTP Error ({e.resp.status}) during comment fetch: {str(e)}")
            # Check for API quota exceeded (403 Forbidden with specific reason)
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                st.error("YouTube API Quota Exceeded! Please check your Google Cloud Console.")
                return [] # No point retrying if quota is hit
            # Handle retryable errors
            elif e.resp.status in [429, 500, 503]:
                wait_time = (2 ** attempt) + random.uniform(0, 1) # Exponential backoff with jitter
                st.warning(f"YouTube API error ({e.resp.status}): Retrying in {wait_time:.1f} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                # For other non-retryable HttpErrors (e.g., 400 Bad Request, 404 Not Found)
                st.error(f"Non-retryable YouTube API error: {str(e)}")
                return []
        except Exception as e:
            # Catch any other unexpected errors during comment fetching
            st.error(f"An unexpected error occurred while fetching comments: {type(e).__name__}: {str(e)}")
            return []

    st.error("Maximum retries exceeded for YouTube API. Could not fetch comments after multiple attempts.")
    return [] # Return empty list if all retries fail

# --- Analyze Sentiment ---
def analyze_sentiment(comments_list, batch_size=32):
    """
    Analyzes the sentiment of a list of comments using the pre-loaded pipeline,
    processing them in batches to manage memory.
    """
    results = []
    # Process comments in batches
    for i in range(0, len(comments_list), batch_size):
        batch = comments_list[i:i + batch_size]
        # Truncate comments to avoid exceeding model's max input length (typically 512 tokens)
        truncated_batch = [comment[:512] for comment in batch]
        
        try:
            # The sentiment_pipeline expects a list of strings
            batch_results = sentiment_pipeline(truncated_batch)
            
            for original_comment, result in zip(batch, batch_results):
                results.append({
                    "comment": original_comment, # Store the original comment text
                    "label": result["label"],    # e.g., 'POSITIVE', 'NEGATIVE'
                    "score": round(result["score"], 3) # Confidence score
                })
        except Exception as e:
            # Log a warning if a batch fails, but continue with other batches
            st.warning(f"Couldn't analyze a batch of comments starting with '{truncated_batch[0][:50]}...': {type(e).__name__}: {str(e)}")
    
    return pd.DataFrame(results)

# --- Streamlit UI (main app logic) ---
st.title("📊 YouTube Video Comment Sentiment Analysis")
st.markdown("Analyze the sentiment of comments on any YouTube video using a DistilBERT model.")

video_url = st.text_input(
    "Enter YouTube Video URL:",
    placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Example URL
)

if st.button("Analyze Comments", type="primary"):
    if not video_url:
        st.warning("Please enter a YouTube URL to analyze.")
    else:
        with st.spinner("Fetching and analyzing comments... This may take a moment."):
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL. Please check the URL format and try again.")
            else:
                comments_data = get_comments_with_likes_filtered(video_id)
                
                if not comments_data:
                    st.error("No comments found or an error occurred while fetching comments. Please check the video ID, your API key/quota, or try again later.")
                else:
                    comment_texts = [c['text'] for c in comments_data]
                    # Moved this line and subsequent display logic inside the button click
                    df_sentiment = analyze_sentiment(comment_texts)

                    if df_sentiment.empty:
                        st.warning("No comments could be analyzed for sentiment. This might happen if all comments were too short or caused model errors.")
                    else:
                        st.success("✅ Analysis Complete!")

                        # --- Sentiment Summary ---
                        st.subheader("🔍 Sentiment Summary")
                        sentiment_counts = df_sentiment['label'].value_counts()
                        pos = sentiment_counts.get("POSITIVE", 0)
                        neg = sentiment_counts.get("NEGATIVE", 0)
                        total = pos + neg

                        if total > 0:
                            pos_percent = round((pos / total) * 100, 1)
                            neg_percent = round((neg / total) * 100, 1)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Comments Analyzed", total)
                            with col2:
                                st.metric("Positive Comments", f"{pos} ({pos_percent}%)")
                            with col3:
                                st.metric("Negative Comments", f"{neg} ({neg_percent}%)")
                            
                            st.markdown("---") # Separator

                            # --- Overall Verdict ---
                            st.subheader("General Sentiment Towards Video:")
                            if pos_percent >= 70:
                                verdict = "## 🎉 Very Positive! People really like this video!"
                                st.balloons() # Add some fun for very positive results
                            elif pos_percent >= 50:
                                verdict = "## 🙂 Mostly Positive. The general sentiment is favorable."
                            else:
                                verdict = "## ⚠️ Mixed to Negative. There might be some critical feedback."
                            st.info(verdict)

                            st.markdown("---") # Separator

                            # --- Most Liked Comment (from original comments_data) ---
                            # Ensure comments_data is not empty before trying to find max
                            if comments_data:
                                # Find the original comment object with the most likes
                                most_liked_comment_data = max(comments_data, key=lambda c: c['likes'])
                                with st.expander("⭐ Most Liked Comment"):
                                    st.write(most_liked_comment_data['text'])
                                    st.caption(f"👍 {most_liked_comment_data['likes']} likes")
                                    # Optionally, get sentiment for the most liked comment
                                    most_liked_sentiment = df_sentiment[df_sentiment['comment'] == most_liked_comment_data['text']]
                                    if not most_liked_sentiment.empty:
                                        sentiment_label = most_liked_sentiment.iloc[0]['label']
                                        sentiment_score = most_liked_sentiment.iloc[0]['score']
                                        st.caption(f"Sentiment: {sentiment_label} (Score: {sentiment_score})")


                            st.markdown("---") # Separator

                            # --- Show All Results (Dataframe) ---
                            st.subheader("📋 All Comments & Sentiments")
                            st.dataframe(df_sentiment, use_container_width=True)

                        else:
                            st.info("No comments with measurable sentiment were found after filtering.")