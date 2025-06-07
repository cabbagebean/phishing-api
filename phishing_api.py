from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from phishing_utils import extract_features  # Import your feature extraction logic

# Define the structure of the incoming request
class EmailData(BaseModel):
    email_text: str
    sender_address: str

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="Detects phishing emails using a trained Random Forest model.",
    version="1.0"
)

# Load the trained model once at startup
try:
    model = joblib.load("phishing_detection_random_tuned.joblib")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to the Phishing Detection API."}

# Prediction endpoint
@app.post("/predict")
def predict(data: EmailData):
    if model is None:
        return {"error": "Model not loaded. Check server logs."}

    try:
        # Extract features from incoming email content
        features = extract_features(data.email_text, data.sender_address)

        # Get prediction
        prediction = model.predict([features])[0]
        label = "phishing" if prediction == 1 else "legit"

        return {"prediction": label}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
