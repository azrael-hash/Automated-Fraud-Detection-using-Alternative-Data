import praw
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, UTC

# ✅ Ensure VADER Lexicon is available
nltk.download('vader_lexicon')

# 🔑 Reddit API Credentials
REDDIT_CLIENT_ID = "YhPMmhcWCHHE2qWMxCeuSQ"
REDDIT_CLIENT_SECRET = "9IOMHddMiBzUraPl4ErrQaf6_qRhww"
REDDIT_USER_AGENT = "fraud_detection"

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

sia = SentimentIntensityAnalyzer()

# 🔎 Search Reddit Posts
def fetch_social_data(query="fraud OR scam", limit=100):
    posts = []
    subreddit = reddit.subreddit("news")

    for post in subreddit.search(query, limit=limit):
        sentiment_score = sia.polarity_scores(post.title)["compound"]
        
        posts.append([
            post.title,
            post.score,  # Upvotes
            post.num_comments,
            post.url,
            datetime.fromtimestamp(post.created_utc, UTC).strftime('%Y-%m-%d %H:%M:%S'),
            sentiment_score  # Sentiment Score
        ])

    df = pd.DataFrame(posts, columns=["Title", "Upvotes", "Comments", "URL", "Post_Age", "Sentiment_Score"])
    return df

# Fetch and save data
df_social = fetch_social_data()
df_social.to_csv("reddit_posts.csv", index=False)
print("✅ Social media data saved successfully!")
