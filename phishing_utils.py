import numpy as np
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
except:
    import nltk
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

# Define phishing-related keywords
phishing_keywords = [
    "verify", "login", "password", "urgent", "account", "update",
    "click", "security", "confirm", "suspend", "limited", "warning",
    "attention", "dear customer", "act now", "bank", "invoice"
]

# Define common free email domains
free_email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]

# === FEATURE EXTRACTION FUNCTION ===
def extract_features(email_text: str, sender_address: str):
    # Clean and lowercase text
    text = email_text.lower()
    sender = sender_address.lower()

    # Feature 1: Log-transformed length of sender address
    sender_length = len(sender)
    log_sender_length = np.log1p(sender_length)

    # Feature 2: Sentiment score of the email content
    sentiment_score = sia.polarity_scores(text)["compound"]

    # Feature 3: Count of phishing keywords
    keyword_count = sum(1 for word in phishing_keywords if word in text)
    
    # Feature 4: Binary flag - does it contain phishing keywords
    contains_keywords = int(keyword_count > 0)

    # Feature 5: Binary flag - is sender from free email provider
    is_free_email = int(any(domain in sender for domain in free_email_domains))

    # === Final feature vector ===
    features = [
        log_sender_length,       # numeric
        sentiment_score,         # numeric
        keyword_count,           # numeric
        contains_keywords,       # binary
        is_free_email            # binary
    ]

    return features
