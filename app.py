# ----------------------------------------------------
# 🌊 DeepReef AI - FastAPI Backend
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
import io
from PIL import Image

# ----------------------------------------------------
# 1️⃣ Initialize FastAPI App
# ----------------------------------------------------
app = FastAPI(title="DeepReef AI Backend", description="Predict coral health and bleaching severity")

# Enable CORS (so your frontend can talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# 2️⃣ Load Trained Models
# ----------------------------------------------------
# CNN Model (MobileNetV2)
cnn_model_path = "coral_mobilenet_best.h5"
cnn_model = tf.keras.models.load_model(cnn_model_path)

# Random Forest Model and Encoder
rf_model_path = "coral_bleaching_model.pkl"       # <-- replace with your RF model filename
encoder_path = "label_encoder.pkl"              # <-- replace with your label encoder filename
train_columns_path = "train_columns.pkl"        # <-- to maintain one-hot column order

rf = joblib.load(rf_model_path)
label_encoder: LabelEncoder = joblib.load(encoder_path)
X_train_columns = joblib.load(train_columns_path)  # list of training columns

# ----------------------------------------------------
# 3️⃣ Utility: Preprocess Coral Image
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
# 4️⃣ Endpoint: Predict Coral Health (Image)
# ----------------------------------------------------
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    try:
        img_array = preprocess_image(file)
        pred = cnn_model.predict(img_array)[0]
        class_labels = ['Bleached Coral', 'Healthy Coral']
        pred_index = np.argmax(pred)
        confidence = float(pred[pred_index])
        label = class_labels[pred_index]

        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------
# 5️⃣ Endpoint: Predict Bleaching Severity (Form Inputs)
# ----------------------------------------------------
@app.post("/predict_severity")
async def predict_severity(
    Temperature_Mean: float = Form(...),
    Windspeed: float = Form(...),
    TSA: float = Form(...),
    Ocean_Name: str = Form(...),
    Exposure: str = Form(...)
):
    try:
        # Create DataFrame
        new_site = pd.DataFrame([{
            'Temperature_Mean': Temperature_Mean,
            'Windspeed': Windspeed,
            'TSA': TSA,
            'Ocean_Name': Ocean_Name,
            'Exposure': Exposure
        }])

        # One-hot encode
        new_encoded = pd.get_dummies(new_site, columns=['Ocean_Name', 'Exposure'], drop_first=True)

        # Add missing columns
        for col in X_train_columns:
            if col not in new_encoded.columns:
                new_encoded[col] = 0

        # Align order
        new_encoded = new_encoded[X_train_columns]

        # Predict
        pred_label = rf.predict(new_encoded)
        pred_class = label_encoder.inverse_transform(pred_label)

        return {
            "bleaching_severity": pred_class[0]
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
        "endpoints": ["/predict_image", "/predict_severity"]
    }

# ----------------------------------------------------
# 7️⃣ Run the Server
# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
