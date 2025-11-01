import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import io
import os
import requests
import mysql.connector
from mysql.connector import Error

# --- Configuration ---
model_path = r'C:\Users\suraj\Desktop\COGNIFIT\food_classifier_model.pth'
nutrition_db_path = r'C:\Users\suraj\Desktop\COGNIFIT\nutrional_data.json'
food_classification_path = r'C:\Users\suraj\Desktop\COGNIFIT\Food Classification'

# Gemini API configuration
apiKey = "AIzaSyCL1lZnvs0WdrpNMMPHelJK6tsYXFg9hUM"
apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# MySQL Database Config
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'Cognifit'
}

# --- Load model and nutrition data ---
class_names = sorted(os.listdir(food_classification_path))
num_classes = len(class_names)

def load_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
    with open(nutrition_db_path, 'r') as f:
        nutrition_data = json.load(f)
    print("✅ Model and nutrition data loaded successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR loading model or nutrition data: {e}")
    raise SystemExit(1)

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper: scale macros based on grams ---
def scale_macros(nutrition, grams):
    try:
        grams = float(grams)
    except ValueError:
        grams = 100.0
    factor = grams / 100
    return {
        "calories": round(nutrition.get("calories", 0) * factor, 2),
        "protein": round(nutrition.get("protein", 0) * factor, 2),
        "carbs": round(nutrition.get("carbs", 0) * factor, 2),
        "fat": round(nutrition.get("fat", 0) * factor, 2)
    }

# --- Helper: save log to MySQL ---
def save_log_to_db(food_name, nutrition_info, grams=100):
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        sql_query = """
        INSERT INTO food_log (food_name, calories, protein, carbs, fat, grams)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        data_tuple = (
            food_name,
            nutrition_info.get('calories'),
            nutrition_info.get('protein'),
            nutrition_info.get('carbs'),
            nutrition_info.get('fat'),
            grams
        )
        cursor.execute(sql_query, data_tuple)
        conn.commit()
        print(f"✅ Logged '{food_name}' ({grams} g) to database.")
    except Error as e:
        print(f"❌ Database Error: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# --- POST: Predict Image ---
@app.post("/predict")
async def predict_food(file: UploadFile = File(...), grams: float = Form(100.0)):
    try:
        grams = float(grams)
        contents = await file.read()
        if not file.content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_preprocessed = preprocess(img)
        input_tensor = img_preprocessed.unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]

        nutrition = nutrition_data.get(predicted_class)
        if not nutrition:
            raise HTTPException(status_code=404, detail="Nutrition data not found for this food item.")

        scaled_nutrition = scale_macros(nutrition, grams)
        save_log_to_db(predicted_class, scaled_nutrition, grams)

        return {
            "predicted_food": predicted_class,
            "nutrition_info": scaled_nutrition,
            "grams": grams
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- POST: Manual Gemini Lookup ---
@app.post("/manual-lookup")
async def manual_lookup(request_data: dict):
    try:
        food_name = request_data.get('food_name')
        grams = float(request_data.get('grams', 100))

        if not food_name:
            raise HTTPException(status_code=400, detail="Missing food_name in request.")
        if not apiKey:
            raise HTTPException(status_code=500, detail="Gemini API key not configured.")

        user_query = f"Provide the calories (kcal), protein (g), fat (g), and carbs (g) for a 100 g serving of {food_name} in JSON format."

        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "calories": {"type": "number"},
                        "protein": {"type": "number"},
                        "fat": {"type": "number"},
                        "carbs": {"type": "number"}
                    },
                }
            }
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{apiUrl}?key={apiKey}", headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Gemini API Error: {response.text}")

        result = response.json()
        ai_text = result['candidates'][0]['content']['parts'][0]['text']
        api_nutrition_data = json.loads(ai_text)

        scaled_nutrition = scale_macros(api_nutrition_data, grams)
        save_log_to_db(food_name, scaled_nutrition, grams)

        return {
            "name": food_name,
            "grams": grams,
            "calories": scaled_nutrition["calories"],
            "protein": scaled_nutrition["protein"],
            "fat": scaled_nutrition["fat"],
            "carbs": scaled_nutrition["carbs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual lookup error: {str(e)}")

# --- GET: Daily Logs ---
@app.get("/logs")
async def get_logs_for_today():
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM food_log WHERE DATE(log_date) = CURDATE() ORDER BY log_date DESC"
        cursor.execute(query)
        logs = cursor.fetchall()
        return {"logs": logs}
    except Error as e:
        print(f"❌ Error fetching logs: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch logs.")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
