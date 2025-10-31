import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import io
import os
import requests
import mysql.connector  # --- ADDED: Import MySQL connector
from mysql.connector import Error

# --- Configuration: Update these paths to your system ---
model_path = r'C:\Users\suraj\Desktop\food_calc\food_classifier_model.pth'
nutrition_db_path = r'C:\Users\suraj\Desktop\food_calc\nutrional_data.json'
food_classification_path = r'C:\Users\suraj\Desktop\food_calc\Food Classification'

# Gemini API configuration
apiKey = "AIzaSyCL1lZnvs0WdrpNMMPHelJK6tsYXFg9hUM"
apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# --- ADDED: Database Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'Cognifit'  # Your database name
}
# -----------------------------------

# Define the list of classes in the same order as they were trained
class_names = sorted(os.listdir(food_classification_path))
num_classes = len(class_names)

# --- Load the trained model into memory ---
def load_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

try:
    model = load_model()
    with open(nutrition_db_path, 'r') as f:
        nutrition_data = json.load(f)
except FileNotFoundError as e:
    print(f"FATAL ERROR: Required file not found during startup: {e}")
    raise SystemExit(1)
except Exception as e:
    print(f"FATAL ERROR: Could not load model or nutrition data: {e}")
    raise SystemExit(1)

# --- Define Image Preprocessing Pipeline ---
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Setup FastAPI Application ---
app = FastAPI()

# --- Configure CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ADDED: Helper function to save logs to the database ---
def save_log_to_db(food_name, nutrition_info):
    """Connects to the MySQL database and inserts a food log entry."""
    conn = None  # Initialize conn to None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        sql_query = """
        INSERT INTO food_log (food_name, calories, protein, carbs, fat)
        VALUES (%s, %s, %s, %s, %s)
        """
        data_tuple = (
            food_name,
            nutrition_info.get('calories'),
            nutrition_info.get('protein'),
            nutrition_info.get('carbs'),
            nutrition_info.get('fat')
        )
        
        cursor.execute(sql_query, data_tuple)
        conn.commit()
        print(f"Successfully logged '{food_name}' to the MySQL database.")
        
    except Error as e:
        print(f"Error saving to MySQL database: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
# -------------------------------------------------------------

# --- POST Endpoint for Image Prediction ---
@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        if not file.content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        img_preprocessed = preprocess(img)
        input_tensor = img_preprocessed.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
        
        nutrition = nutrition_data.get(predicted_class, {"error": "Local nutrition data not found."})
        
        # --- MODIFIED: Save the log to the database ---
        if "error" not in nutrition:
            save_log_to_db(predicted_class, nutrition)
        # --------------------------------------------
            
        response = {
            "predicted_food": predicted_class,
            "nutrition_info": nutrition
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# --- Endpoint for AI-Powered Manual Lookup ---
@app.post("/manual-lookup")
async def manual_lookup(food_name: dict):
    try:
        if not apiKey:
            raise HTTPException(status_code=500, detail="Gemini API key is not configured.")

        user_query = f"Provide the calories (kcal), protein (g), fat (g), and carbs (g) for a 100g serving of {food_name['food_name']} in JSON format. Only provide the JSON object, do not include any other text, explanation, or backticks."
        
        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "calories": {"type": "number"}, "protein": {"type": "number"},
                        "fat": {"type": "number"}, "carbs": {"type": "number"}
                    },
                }
            }
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{apiUrl}?key={apiKey}", headers=headers, data=json.dumps(payload))
        
        if response.status_code != 200:
            error_detail = response.text if response.text else "Unknown error."
            raise HTTPException(status_code=response.status_code, detail=f"Gemini API request failed. Status: {response.status_code}. Detail: {error_detail}")

        result = response.json()
        ai_response_text = result['candidates'][0]['content']['parts'][0]['text']
        api_nutrition_data = json.loads(ai_response_text)

        # --- MODIFIED: Save the manual log to the database ---
        save_log_to_db(food_name['food_name'], api_nutrition_data)
        # ----------------------------------------------------

        return {
            "calories": api_nutrition_data.get("calories", 0),
            "protein": api_nutrition_data.get("protein", 0),
            "fat": api_nutrition_data.get("fat", 0),
            "carbs": api_nutrition_data.get("carbs", 0),
            "name": food_name['food_name']
        }
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse JSON response from AI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- ADDED: New endpoint to fetch daily logs ---
@app.get("/logs")
async def get_logs_for_today():
    """Fetches all food logs for the current day from MySQL."""
    logs = []
    conn = None  # Initialize conn to None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True) # dictionary=True returns rows as dicts

        # CURDATE() is the MySQL function to get the current date
        query = "SELECT * FROM food_log WHERE DATE(log_date) = CURDATE() ORDER BY log_date DESC"
        cursor.execute(query)
        
        logs = cursor.fetchall()

    except Error as e:
        print(f"Error fetching logs from MySQL: {e}")
        # Optionally, raise an HTTPException to inform the client
        raise HTTPException(status_code=500, detail="Could not fetch logs from the database.")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            
    return {"logs": logs}
# ---------------------------------------------