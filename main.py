import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import json
import io
import os
import requests
import mysql.connector 
from mysql.connector import Error
from pydantic import BaseModel, Field
from typing import Optional

# --- Configuration: Update these paths to your system ---
# NOTE: These paths MUST be correct for your ML assets
model_path = r'C:\Users\suraj\Desktop\COGNIFIT\food_classifier_model.pth' 
nutrition_db_path = r'C:\Users\suraj\Desktop\COGNIFIT\nutrional_data.json'
food_classification_path = r'C:\Users\suraj\Desktop\COGNIFIT\Food Classification' 

# Gemini API configuration
apiKey = "AIzaSyCL1lZnvs0WdrpNMMPHelJK6tsYXFg9hUM"
apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# --- Database Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'cognifit' 
}
# -----------------------------------

# Pydantic Model for incoming and outgoing data
class MealLogEntry(BaseModel):
    user_id: int
    food_name: str
    calories: float
    protein: float
    fat: float
    carbs: float
    grams: float = Field(..., alias='grams') # Maps JS 'grams' to quantity_value
    unit: str
    log_date: str

# --- Load ML Assets ---
try:
    # 1. Load Classes
    # CRITICAL FIX: Ensure path resolution for classes folder
    class_names = sorted(os.listdir(food_classification_path))
    num_classes = len(class_names)
    
    # 2. Load Model
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters(): param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 3. Load Nutrition Data
    with open(nutrition_db_path, 'r') as f:
        nutrition_data = json.load(f)
        
    print(f"✅ Food Model and Data loaded successfully for {num_classes} classes.")

except Exception as e:
    # This block prevents the server from crashing immediately if ML assets are missing
    print(f"❌ CRITICAL WARNING: ML asset loading failed ({e}). Image Prediction disabled.")
    model = None
    nutrition_data = {}
    class_names = []


# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Setup FastAPI Application ---
app = FastAPI()

# --- Configure CORS middleware (CRITICAL for Flask to talk to FastAPI) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Database Helper Functions (Personalized)
# -------------------------
def save_log_to_db_internal(meal: MealLogEntry):
    """Internal function to save meal to database."""
    conn = None 
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # --- FIXED QUERY TO MATCH THE 9 COLUMNS IN YOUR food_log TABLE ---
        sql_query = """
        INSERT INTO food_log (user_id, food_name, calories, protein, fat, carbs, grams, log_date)
        VALUES ( %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data_tuple = (
            meal.user_id, meal.food_name, meal.calories, meal.protein, meal.fat, meal.carbs,
            meal.grams,  meal.log_date
        )
        
        cursor.execute(sql_query, data_tuple)
        conn.commit()
        return True
    except Error as e:
        print(f"❌ Database Write Error: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def get_logs_by_user_and_date(user_id: int, log_date: str):
    """Fetches all food logs for a specific date and user."""
    conn = None
    logs = []
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True) 
        query = "SELECT id, food_name, calories, protein, fat, carbs, grams, unit, log_date FROM food_log WHERE user_id = %s AND log_date = %s ORDER BY created_at DESC"
        cursor.execute(query, (user_id, log_date,))
        logs = cursor.fetchall()
        return logs
    except Error as e:
        print(f"❌ Database Read Error: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# -------------------------
# API Endpoints
# -------------------------

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    """Handles image upload and runs ML model to return 100g data."""
    if not model:
         raise HTTPException(status_code=500, detail="Image prediction service is unavailable. Model failed to load.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        if not file.content_type or not file.content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        # ML Prediction Logic...
        img_preprocessed = preprocess(img)
        input_tensor = img_preprocessed.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(probabilities).item()
        
        # --- CRITICAL FIX: Confidence Threshold Filter ---
        CONFIDENCE_THRESHOLD = 0.60 # 60%
        if confidence < CONFIDENCE_THRESHOLD:
            # Tell the client that the prediction failed due to uncertainty
            raise HTTPException(
                status_code=404, 
                detail="UNCERTAIN_PREDICTION_FALLBACK_REQUIRED"
            )
        
        # Standard successful prediction if confidence is high
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
        
        nutrition = nutrition_data.get(predicted_class, {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0})
        
        return {
            "predicted_food": predicted_class,
            "nutrition_info": nutrition # Returns 100g data
        }
    
    except HTTPException:
        # Re-raise the 404 from the confidence check
        raise
    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
@app.post("/manual-lookup")
async def manual_lookup(request_data: dict):
    """Uses Gemini API to get 100g nutritional data for manual entry."""
    try:
        food_name = request_data.get('food_name')
        if not food_name:
            raise HTTPException(status_code=400, detail="Missing food name.")
        if not apiKey:
            raise HTTPException(status_code=500, detail="Gemini API key is not configured.")

        # Gemini API Request Logic (Returns 100g JSON)
        user_query = f"Provide the calories (kcal), protein (g), fat (g), and carbs (g) for a 100g serving of {food_name} in JSON format. Only provide the JSON object, do not include any other text, explanation, or backticks."
        
        payload = {"contents": [{"parts": [{"text": user_query}]}], "generationConfig": {"responseMimeType": "application/json", "responseSchema": {"type": "OBJECT", "properties": {"calories": {"type": "number"}, "protein": {"type": "number"}, "fat": {"type": "number"}, "carbs": {"type": "number"}}}}}
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{apiUrl}?key={apiKey}", headers=headers, data=json.dumps(payload))
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Gemini API Error: {response.text}")

        result = response.json()
        ai_response_text = result['candidates'][0]['content']['parts'][0]['text']
        api_nutrition_data = json.loads(ai_response_text)

        # Returns flat 100g data required by frontend
        return {
            "calories": api_nutrition_data.get("calories", 0),
            "protein": api_nutrition_data.get("protein", 0),
            "fat": api_nutrition_data.get("fat", 0),
            "carbs": api_nutrition_data.get("carbs", 0),
            "name": food_name
        }
    
    except Exception as e:
        print(f"❌ Error in /manual-lookup: {e}")
        raise HTTPException(status_code=500, detail=f"Manual lookup error: {str(e)}")

@app.post("/add-meal-to-db")
async def add_meal_to_db(meal: MealLogEntry):
    """Saves a fully calculated meal entry to the food_log table."""
    # The frontend is responsible for calculating scaled macros and including user_id
    if save_log_to_db_internal(meal):
        return {"success": True, "message": "Meal logged successfully."}
    raise HTTPException(status_code=500, detail="Database insertion failed.")

@app.get("/get-meals/{user_id}/{log_date}")
async def get_meals(user_id: int, log_date: str):
    """Fetches all food logs for a specific date and user."""
    logs = get_logs_by_user_and_date(user_id, log_date)
    return {"logs": logs}

# -------------------------
# Run the application (for standalone testing)
# -------------------------
if __name__ == '__main__':
    import uvicorn
    # CRITICAL: Runs on a separate port (8001) for the hybrid architecture.
    uvicorn.run(app, host="0.0.0.0", port=8001)
