# ----------------------------------------------------
# 🌊 DeepReef AI - Unified FastAPI Backend
# ----------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import joblib
from PIL import Image
import io
import google.generativeai as genai

# ----------------------------------------------------
# 1️⃣ Configure Gemini API (keep key here for now)
# ----------------------------------------------------
GEMINI_API_KEY = "AIzaSyBwEQJ3mlHKD0yE51z0gBI90F6zhnOrpyg"  # <-- 🔒 Replace with your actual key
genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------------------------------
# 2️⃣ Initialize FastAPI App
# ----------------------------------------------------
app = FastAPI(title="DeepReef AI Backend", description="Unified endpoint for coral health and bleaching risk")

# Enable CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# 3️⃣ Load Models
# ----------------------------------------------------
# CNN Model for Coral Health Classification
cnn_model_path = "coral_mobilenet_best.h5"
cnn_model = tf.keras.models.load_model(cnn_model_path)

# Random Forest Model for Bleaching Severity
rf_model_path = "coral_bleaching_model.pkl"
encoder_path = "label_encoder.pkl"
train_columns_path = "train_columns.pkl"

rf = joblib.load(rf_model_path)
label_encoder: LabelEncoder = joblib.load(encoder_path)
X_train_columns = joblib.load(train_columns_path)

# ----------------------------------------------------
# 4️⃣ Image Preprocessing Utility
# ----------------------------------------------------
def preprocess_image(uploaded_file):
    TARGET_SIZE = (160, 160)
    image_bytes = uploaded_file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# ----------------------------------------------------
# 5️⃣ Unified Endpoint for Coral Analysis
# ----------------------------------------------------
@app.post("/analyze_coral")
async def analyze_coral(
    file: UploadFile = File(...),
    Temperature_Mean: float = Form(...),
    Windspeed: float = Form(...),
    TSA: float = Form(...),
    Ocean_Name: str = Form(...),
    Exposure: str = Form(...)
):
    try:
        # ----- Step 1: CNN Prediction -----
        img_array = preprocess_image(file)
        pred = cnn_model.predict(img_array)[0]
        class_labels = ['Bleached Coral', 'Healthy Coral']
        pred_index = np.argmax(pred)
        confidence = float(pred[pred_index])
        coral_status = class_labels[pred_index]

        # ----- Step 2: Random Forest Prediction -----
        new_site = pd.DataFrame([{
            'Temperature_Mean': Temperature_Mean,
            'Windspeed': Windspeed,
            'TSA': TSA,
            'Ocean_Name': Ocean_Name,
            'Exposure': Exposure
        }])

        new_encoded = pd.get_dummies(new_site, columns=['Ocean_Name', 'Exposure'], drop_first=True)

        # Add missing columns
        for col in X_train_columns:
            if col not in new_encoded.columns:
                new_encoded[col] = 0

        # Reorder columns to match training data
        new_encoded = new_encoded[X_train_columns]

        # Predict severity
        severity_pred = rf.predict(new_encoded)
        severity = label_encoder.inverse_transform(severity_pred)[0]

        # ----- Step 3: Compose Gemini Prompt -----
        prompt = f"""
        The coral image was analyzed using AI.
        Image inference: {coral_status} (confidence: {confidence*100:.2f}%)
        Environmental parameters:
        - Temperature_Mean: {Temperature_Mean}
        - Windspeed: {Windspeed}
        - TSA: {TSA}
        - Ocean_Name: {Ocean_Name}
        - Exposure: {Exposure}
        Random Forest predicted bleaching severity as: {severity}.

        Based on these results, write a concise 3–4 sentence scientific summary
        explaining the coral’s health status, probable causes, and risk level.
        """

        # ----- Step 4: Query Gemini -----
        model = genai.GenerativeModel("gemini-1.5-flash")
        gemini_response = model.generate_content(prompt)
        summary = gemini_response.text

        # ----- Step 5: Return Results -----
        return {
            "image_prediction": coral_status,
            "confidence": round(confidence * 100, 2),
            "bleaching_severity": severity,
            "gemini_summary": summary
        }

    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------
# 6️⃣ Root Endpoint
# ----------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome to DeepReef AI 🌊",
        "endpoint": "/analyze_coral",
        "instructions": "POST an image + environmental parameters to get predictions + Gemini summary."
    }

# ----------------------------------------------------
# 7️⃣ Run Server
# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
