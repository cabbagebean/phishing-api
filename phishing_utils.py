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

def extract_features(email_text: str, sender_address: str):
    # Clean and lowercase text
    text = email_text.lower()
    sender = sender_address.lower()

    # Feature 1: Log-transformed length of sender address
    sender_length = len(sender)
    log_sender_length = np.log1p(sender_length)

    # Feature 2: Sentiment score
    sentiment_score = sia.polarity_scores(text)["compound"]

    # Feature 3: Count of phishing keywords
    keyword_count = sum(1 for word in phishing_keywords if word in text)

    # Feature 4: Binary - contains phishing keywords
    contains_keywords = int(keyword_count > 0)

    # Feature 5: Binary - is sender from free email domain
    is_free_email = int(any(domain in sender for domain in free_email_domains))

    # Feature 6: Binary - is disposable email domain
    disposable_domains = ["mailinator.com", "10minutemail.com", "trashmail.com"]
    is_disposable_email = int(any(domain in sender for domain in disposable_domains))

    # Feature 7: Binary - suspicious characters in sender
    has_suspicious_chars = int(bool(re.search(r"[!$%^*#~]", sender)))

    # Feature 8: Binary - contains URL
    has_url = int("http" in text or "www." in text or ".com" in text)

    # Feature 9: Interaction feature
    url_x_keyword = keyword_count * has_url

    # Return as list in the exact order used during model training
    return [
        email_text,              # This is passed to the TfidfVectorizer
        log_sender_length,
        sentiment_score,
        keyword_count,
        contains_keywords,
        is_free_email,
        is_disposable_email,
        has_suspicious_chars,
        url_x_keyword
    ]
