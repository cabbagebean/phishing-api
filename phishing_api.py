from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
import pandas as pd
import gdown
import numpy as np
from phishing_utils import extract_features  # Your custom feature extraction

# Define the structure of incoming data
class EmailData(BaseModel):
    email_text: str
    sender_address: str

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="Detects phishing emails using a trained Random Forest model.",
    version="1.0"
)

# === Model loading configuration ===
MODEL_DIR = "model"
MODEL_FILENAME = "phishing_detection_random_tuned.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Google Drive File ID
GOOGLE_DRIVE_FILE_ID = "1UbPPC3XoxMuOeHp0Rfa4QVCNJL0FUpr-"

def download_model_from_drive(file_id: str, destination_path: str):
    print("📥 Downloading model using gdown...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination_path, quiet=False)
    print("✅ Model downloaded successfully.")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the model
try:
    if not os.path.exists(MODEL_PATH):
        download_model_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Phishing Detection API."}

# Prediction endpoint
@app.post("/predict")
def predict(data: EmailData):
    if model is None:
        return {"error": "Model not loaded. Check server logs."}

    try:
        # Extract features from the incoming email
        features = extract_features(data.email_text, data.sender_address)

        # Convert to DataFrame to preserve column names
        input_df = pd.DataFrame([features])

        # Generate prediction
        prediction = model.predict(input_df)[0]
        label = "phishing" if prediction == 1 else "legit"

        return {"prediction": label}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}