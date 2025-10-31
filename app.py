from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import mysql.connector
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import re
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for login sessions

# Standard MySQL database configuration (platform independent)
db_cognifit = {
    "host": "localhost",
    "user": "root",
    "password": "",  # Empty password by default
    "database": "cognifit"
    # Removed port - will use default MySQL port 3306
}

# ML Model Storage
ML_MODEL_PATH = "ml_models/cycle_predictor.pkl"
SCALER_PATH = "ml_models/scaler.pkl"

# -------------------------
# Database Connection Helper
# -------------------------
def get_db_connection():
    """Create and return database connection with error handling"""
    try:
        conn = mysql.connector.connect(**db_cognifit)
        return conn
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

# -------------------------
# Database Initialization
# -------------------------
def init_database():
    """Initialize database with required columns and tables"""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database during initialization")
        return
    
    try:
        cur = conn.cursor()
        
        # Check if phone_number column exists
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'users' 
            AND COLUMN_NAME = 'phone_number'
        """, (db_cognifit["database"],))
        column_exists = cur.fetchone()[0]
        
        if not column_exists:
            # Add phone_number column if it doesn't exist
            cur.execute("ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)")
            print("Added phone_number column to users table")
        
        # Check if calendar_notes table exists
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'calendar_notes'
        """, (db_cognifit["database"],))
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            # Create calendar_notes table
            cur.execute("""
                CREATE TABLE calendar_notes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    note_date DATE NOT NULL,
                    note_content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user_date (user_id, note_date)
                )
            """)
            print("Created calendar_notes table")
        
        # Check if blogs table exists
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'blogs'
        """, (db_cognifit["database"],))
        blogs_table_exists = cur.fetchone()[0]
        
        if not blogs_table_exists:
            # Create blogs table
            cur.execute("""
                CREATE TABLE blogs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    excerpt TEXT,
                    category VARCHAR(100) NOT NULL,
                    category_color VARCHAR(50) DEFAULT '#abc85f',
                    author VARCHAR(100) NOT NULL,
                    read_time VARCHAR(20) NOT NULL,
                    image_url VARCHAR(500),
                    is_featured BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    status ENUM('published', 'draft') DEFAULT 'published',
                    INDEX idx_category (category),
                    INDEX idx_featured (is_featured),
                    INDEX idx_status (status)
                )
            """)
            print("Created blogs table")
        
        # Check if recipes table exists
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'recipes'
        """, (db_cognifit["database"],))
        recipes_table_exists = cur.fetchone()[0]
        
        if not recipes_table_exists:
            # Create recipes table
            cur.execute("""
                CREATE TABLE recipes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    prep_time VARCHAR(50),
                    cook_time VARCHAR(50),
                    total_time VARCHAR(50) NOT NULL,
                    difficulty VARCHAR(50),
                    calories VARCHAR(50),
                    servings VARCHAR(50),
                    tags TEXT,
                    image_url VARCHAR(500),
                    ingredients TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    is_featured BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_category (category),
                    INDEX idx_featured (is_featured)
                )
            """)
            print("Created recipes table")
        
        conn.commit()
        cur.close()
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()

# Initialize database when app starts
init_database()

# -------------------------
# ML FUNCTIONS
# -------------------------

def load_ml_model():
    """Load trained ML model and scaler"""
    try:
        if os.path.exists(ML_MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(ML_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
    except Exception as e:
        print(f"Error loading ML model: {e}")
    return None, None

def train_ml_model():
    """Train ML model using your dataset"""
    try:
        # Create sample training data from your dataset pattern
        np.random.seed(42)
        n_samples = 100
        
        # Features: [last_cycle_length, cycle_std, cycle_count, age_group]
        X = np.column_stack([
            np.random.randint(25, 35, n_samples),  # last_cycle_length
            np.random.uniform(0, 5, n_samples),    # cycle_std
            np.random.randint(1, 20, n_samples),   # cycle_count
            np.random.randint(1, 5, n_samples)     # age_group
        ])
        
        # Target: next cycle length (based on patterns from your data)
        y = X[:, 0] + np.random.normal(0, 2, n_samples)  # next cycle similar to last + noise
        y = np.clip(y, 21, 35)  # keep within realistic range
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled, y)
        
        # Save model and scaler
        os.makedirs("ml_models", exist_ok=True)
        with open(ML_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        print("ML model trained successfully!")
        return True
        
    except Exception as e:
        print(f"Error training ML model: {e}")
        return False

def predict_next_cycle_ml(user_id, last_period, cycle_length, period_duration, age=None):
    """Predict next cycle using ML"""
    model, scaler = load_ml_model()
    
    if model and scaler:
        try:
            # Get user's historical data
            historical_data = get_user_cycle_history(user_id)
            
            if historical_data and len(historical_data) >= 2:
                # Prepare features
                cycle_lengths = [entry['cycle_length'] for entry in historical_data]
                features = [
                    cycle_length,                    # last_cycle_length
                    np.std(cycle_lengths) if len(cycle_lengths) > 1 else 0,  # cycle_std
                    len(historical_data),           # cycle_count
                    (age // 10) if age else 3       # age_group
                ]
                
                features_scaled = scaler.transform([features])
                predicted_cycle_length = model.predict(features_scaled)[0]
                predicted_cycle_length = max(21, min(35, predicted_cycle_length))
                
                # Calculate dates
                next_period = last_period + timedelta(days=predicted_cycle_length)
                ovulation_day = last_period + timedelta(days=(predicted_cycle_length - 14))
                fertile_start = ovulation_day - timedelta(days=4)
                fertile_end = ovulation_day + timedelta(days=1)
                
                return {
                    'next_period': next_period,
                    'ovulation_day': ovulation_day,
                    'fertile_window': (fertile_start, fertile_end),
                    'predicted_cycle_length': int(predicted_cycle_length),
                    'method': 'ml'
                }
        except Exception as e:
            print(f"ML prediction failed: {e}")
    
    # Fallback to traditional calculation
    return predict_traditional(last_period, cycle_length)

def predict_traditional(last_period, cycle_length):
    """Traditional prediction method"""
    next_period = last_period + timedelta(days=cycle_length)
    ovulation_day = last_period + timedelta(days=(cycle_length - 14))
    fertile_start = ovulation_day - timedelta(days=4)
    fertile_end = ovulation_day + timedelta(days=1)
    
    return {
        'next_period': next_period,
        'ovulation_day': ovulation_day,
        'fertile_window': (fertile_start, fertile_end),
        'predicted_cycle_length': cycle_length,
        'method': 'traditional'
    }

def get_user_cycle_history(user_id):
    """Get user's historical cycle data"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT cycle_length, period_duration, last_period, created_at
            FROM menstrual_cycle 
            WHERE user_id = %s 
            ORDER BY created_at DESC
            LIMIT 10
        """, (user_id,))
        history = cur.fetchall()
        cur.close()
        return history
    except Exception as e:
        print(f"Error getting user history: {e}")
        return []
    finally:
        conn.close()

def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI given weight in kg and height in cm"""
    if not weight_kg or not height_cm:
        return None
    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)
    return round(bmi, 1)

def get_bmi_category(bmi):
    """Get BMI category based on calculated BMI"""
    if bmi is None:
        return "Unknown"
    elif bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# -------------------------
# Routes
# -------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/snap')
def snap():
    return render_template('snap.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    files = {'file': (file.filename, file.read(), file.content_type)}
    
    # Send the image to FastAPI
    response = requests.post(FASTAPI_URL, files=files)
    
    if response.status_code == 200:
        result = response.json()
        return jsonify(result)
    else:
        return jsonify({"error": f"API request failed with {response.status_code}"})

# -------------------------
# Recipe Routes
# -------------------------

@app.route('/recipes')
def recipes():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    if not conn:
        flash("Database connection error", "error")
        return render_template('recipes.html', 
                             user_name=session.get('username'),
                             featured_recipe=None,
                             recipes=[])
    
    try:
        cur = conn.cursor(dictionary=True)
        
        # Get featured recipe
        cur.execute("""
            SELECT * FROM recipes 
            WHERE is_featured = TRUE 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        featured_recipe = cur.fetchone()
        
        # Get other recipes
        cur.execute("""
            SELECT * FROM recipes 
            WHERE (is_featured = FALSE OR is_featured IS NULL) 
            ORDER BY created_at DESC 
            LIMIT 15
        """)
        recipes = cur.fetchall()
        
        cur.close()
        
        return render_template('recipes.html', 
                             user_name=session.get('username'),
                             featured_recipe=featured_recipe,
                             recipes=recipes)
        
    except Exception as e:
        print(f"Error loading recipes: {e}")
        return render_template('recipes.html', 
                             user_name=session.get('username'),
                             featured_recipe=None,
                             recipes=[])
    finally:
        conn.close()

@app.route('/get_recipe/<int:recipe_id>')
def get_recipe(recipe_id):
    """Get single recipe for modal"""
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor(dictionary=True)
        
        cur.execute("SELECT * FROM recipes WHERE id = %s", (recipe_id,))
        recipe = cur.fetchone()
        
        cur.close()
        
        if recipe:
            return jsonify({
                "success": True,
                "recipe": {
                    "id": recipe['id'],
                    "title": recipe['title'],
                    "category": recipe['category'],
                    "prep_time": recipe['prep_time'],
                    "cook_time": recipe['cook_time'],
                    "total_time": recipe['total_time'],
                    "difficulty": recipe['difficulty'],
                    "calories": recipe['calories'],
                    "servings": recipe['servings'],
                    "tags": recipe['tags'],
                    "image_url": recipe['image_url'],
                    "ingredients": recipe['ingredients'],
                    "instructions": recipe['instructions'],
                    "is_featured": recipe['is_featured'],
                    "created_at": recipe['created_at'].strftime('%b %d, %Y') if recipe['created_at'] else ''
                }
            })
        else:
            return jsonify({"success": False, "message": "Recipe not found"}), 404
            
    except Exception as e:
        print(f"Error fetching recipe: {e}")
        return jsonify({"success": False, "message": "Error fetching recipe"}), 500
    finally:
        conn.close()

@app.route('/admin/recipes')
def admin_recipes():
    """Admin panel for managing recipes"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('recipes_admin.html', user_name=session.get('username'))

@app.route('/admin/create_recipe', methods=['POST'])
def create_recipe():
    """Create a new recipe"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        data = request.get_json()
        
        title = data.get('title')
        category = data.get('category')
        prep_time = data.get('prep_time')
        cook_time = data.get('cook_time')
        total_time = data.get('total_time')
        difficulty = data.get('difficulty')
        calories = data.get('calories')
        servings = data.get('servings')
        tags = data.get('tags', '')
        image_url = data.get('image_url', '')
        ingredients = data.get('ingredients')
        instructions = data.get('instructions')
        is_featured = data.get('is_featured', False)
        
        # Validate required fields
        if not all([title, category, total_time, calories, servings, ingredients, instructions]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400
        
        cur = conn.cursor()
        
        # If setting as featured, unfeature other recipes
        if is_featured:
            cur.execute("UPDATE recipes SET is_featured = FALSE WHERE is_featured = TRUE")
        
        # Insert new recipe
        cur.execute("""
            INSERT INTO recipes (title, category, prep_time, cook_time, total_time, difficulty, calories, servings, tags, image_url, ingredients, instructions, is_featured)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (title, category, prep_time, cook_time, total_time, difficulty, calories, servings, tags, image_url, ingredients, instructions, is_featured))
        
        conn.commit()
        recipe_id = cur.lastrowid
        
        cur.close()
        
        return jsonify({
            "success": True, 
            "message": "Recipe created successfully!",
            "recipe_id": recipe_id
        })
        
    except Exception as e:
        print(f"Error creating recipe: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error creating recipe"}), 500
    finally:
        conn.close()

@app.route('/admin/get_recipes')
def admin_get_recipes():
    """Get all recipes for admin panel"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor(dictionary=True)
        
        cur.execute("SELECT * FROM recipes ORDER BY created_at DESC")
        recipes = cur.fetchall()
        
        cur.close()
        
        # Convert datetime objects to strings
        for recipe in recipes:
            if recipe['created_at']:
                recipe['created_at'] = recipe['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if recipe['updated_at']:
                recipe['updated_at'] = recipe['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({"success": True, "recipes": recipes})
        
    except Exception as e:
        print(f"Error fetching recipes: {e}")
        return jsonify({"success": False, "message": "Error fetching recipes"}), 500
    finally:
        conn.close()

@app.route('/admin/update_recipe/<int:recipe_id>', methods=['POST'])
def update_recipe(recipe_id):
    """Update a recipe"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        data = request.get_json()
        
        cur = conn.cursor()
        
        # Build dynamic update query
        update_fields = []
        update_values = []
        
        if 'title' in data:
            update_fields.append("title = %s")
            update_values.append(data['title'])
        if 'category' in data:
            update_fields.append("category = %s")
            update_values.append(data['category'])
        if 'prep_time' in data:
            update_fields.append("prep_time = %s")
            update_values.append(data['prep_time'])
        if 'cook_time' in data:
            update_fields.append("cook_time = %s")
            update_values.append(data['cook_time'])
        if 'total_time' in data:
            update_fields.append("total_time = %s")
            update_values.append(data['total_time'])
        if 'difficulty' in data:
            update_fields.append("difficulty = %s")
            update_values.append(data['difficulty'])
        if 'calories' in data:
            update_fields.append("calories = %s")
            update_values.append(data['calories'])
        if 'servings' in data:
            update_fields.append("servings = %s")
            update_values.append(data['servings'])
        if 'tags' in data:
            update_fields.append("tags = %s")
            update_values.append(data['tags'])
        if 'image_url' in data:
            update_fields.append("image_url = %s")
            update_values.append(data['image_url'])
        if 'ingredients' in data:
            update_fields.append("ingredients = %s")
            update_values.append(data['ingredients'])
        if 'instructions' in data:
            update_fields.append("instructions = %s")
            update_values.append(data['instructions'])
        if 'is_featured' in data:
            update_fields.append("is_featured = %s")
            update_values.append(data['is_featured'])
            # If setting as featured, unfeature other recipes
            if data['is_featured']:
                cur.execute("UPDATE recipes SET is_featured = FALSE WHERE id != %s", (recipe_id,))
        
        if update_fields:
            update_values.append(recipe_id)
            query = f"UPDATE recipes SET {', '.join(update_fields)} WHERE id = %s"
            cur.execute(query, update_values)
            conn.commit()
        
        cur.close()
        
        return jsonify({"success": True, "message": "Recipe updated successfully!"})
        
    except Exception as e:
        print(f"Error updating recipe: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error updating recipe"}), 500
    finally:
        conn.close()

@app.route('/admin/delete_recipe/<int:recipe_id>', methods=['DELETE'])
def delete_recipe(recipe_id):
    """Delete a recipe"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor()
        
        cur.execute("DELETE FROM recipes WHERE id = %s", (recipe_id,))
        conn.commit()
        
        cur.close()
        
        return jsonify({"success": True, "message": "Recipe deleted successfully!"})
        
    except Exception as e:
        print(f"Error deleting recipe: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error deleting recipe"}), 500
    finally:
        conn.close()

@app.route('/blogs')
def blogs():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    if not conn:
        flash("Database connection error", "error")
        return render_template('blogs.html', 
                             user_name=session.get('username'),
                             featured_blog=None,
                             blogs=[])
    
    try:
        cur = conn.cursor(dictionary=True)
        
        # Get featured blog
        cur.execute("""
            SELECT * FROM blogs 
            WHERE is_featured = TRUE AND status = 'published' 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        featured_blog = cur.fetchone()
        
        # Get other blogs
        cur.execute("""
            SELECT * FROM blogs 
            WHERE (is_featured = FALSE OR is_featured IS NULL) AND status = 'published' 
            ORDER BY created_at DESC 
            LIMIT 12
        """)
        blogs = cur.fetchall()
        
        cur.close()
        
        return render_template('blogs.html', 
                             user_name=session.get('username'),
                             featured_blog=featured_blog,
                             blogs=blogs)
        
    except Exception as e:
        print(f"Error loading blogs: {e}")
        return render_template('blogs.html', 
                             user_name=session.get('username'),
                             featured_blog=None,
                             blogs=[])
    finally:
        conn.close()

@app.route('/about')
def about():
    return "About page"

@app.route("/setup", methods=["GET", "POST"])
def setup():
    return render_template("setup.html")

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_data = None
    bmi = None
    bmi_category = None
    
    conn = get_db_connection()
    if conn:
        try:
            # Get user onboarding data including height and weight
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT age, height, weight, activity_level, goals 
                FROM user_onboarding 
                WHERE user_id = %s
            """, (user_id,))
            user_data = cur.fetchone()
            cur.close()
            
            # Calculate BMI if height and weight are available
            if user_data and user_data.get('height') and user_data.get('weight'):
                bmi = calculate_bmi(user_data['weight'], user_data['height'])
                bmi_category = get_bmi_category(bmi)
                
        except mysql.connector.Error as err:
            print(f"Error fetching user data for dashboard: {err}")
            flash("Error loading your health data", "error")
        finally:
            conn.close()
    
    return render_template('dashboard.html', 
                         user_name=session.get('username'),
                         user_data=user_data,
                         bmi=bmi,
                         bmi_category=bmi_category)

# -------------------------
# Onboarding Route
# -------------------------
@app.route('/onboarding', methods=['GET', 'POST'])
def onboarding():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        # Render the onboarding form
        return render_template('onboard.html', user_name=session.get('username'))
    
    elif request.method == 'POST':
        # Handle the onboarding form submission
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": False, "message": "Database connection error"}), 500
        
        try:
            # Get JSON data from the request
            data = request.get_json()
            
            user_id = session['user_id']
            goals = data.get('goals', [])
            activity_level = data.get('activity', '')
            age = data.get('age')
            height = data.get('height')
            weight = data.get('weight')
            injury = data.get('injury', '')
            
            # Convert goals list to string for database storage
            goals_str = ','.join(goals) if goals else ''
            
            cur = conn.cursor()
            
            # Check if user already has onboarding data
            cur.execute("SELECT id FROM user_onboarding WHERE user_id = %s", (user_id,))
            existing_record = cur.fetchone()
            
            if existing_record:
                # Update existing record
                cur.execute("""
                    UPDATE user_onboarding 
                    SET goals = %s, activity_level = %s, age = %s, height = %s, weight = %s, injury_conditions = %s, updated_at = %s
                    WHERE user_id = %s
                """, (goals_str, activity_level, age, height, weight, injury, datetime.now(), user_id))
            else:
                # Insert new record
                cur.execute("""
                    INSERT INTO user_onboarding (user_id, goals, activity_level, age, height, weight, injury_conditions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (user_id, goals_str, activity_level, age, height, weight, injury))
            
            conn.commit()
            cur.close()
            
            return jsonify({"success": True, "message": "Onboarding data saved successfully!"})
            
        except Exception as e:
            print(f"Error saving onboarding data: {e}")
            conn.rollback()
            return jsonify({"success": False, "message": "Error saving data"}), 500
        finally:
            conn.close()

# -------------------------
# Signup
# -------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        gender = request.form['gender']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        # ✅ Allow only Gmail addresses
        gmail_pattern = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
        if not re.match(gmail_pattern, email):
            flash("Only Gmail addresses are allowed!", "error")
            return redirect(url_for("signup"))

        # ✅ Password validation
        if len(password) < 8 or not re.search(r"\d", password) or not re.search(r"[A-Z]", password):
            flash("Password must be at least 8 characters long, include a number and an uppercase letter.", "error")
            return redirect(url_for("signup"))

        # ✅ Password confirmation
        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup"))

        hashed_pw = generate_password_hash(password)

        conn = get_db_connection()
        if not conn:
            flash("Database connection error", "error")
            return redirect(url_for('signup'))

        try:
            cur = conn.cursor()

            # Check if email already exists
            cur.execute("SELECT * FROM users WHERE email=%s", (email,))
            if cur.fetchone():
                flash("Email already registered. Please login.", "error")
                cur.close()
                conn.close()
                return redirect(url_for('login'))

            # Insert new user
            cur.execute("""
                INSERT INTO users (firstname, lastname, gender, email, password)
                VALUES (%s, %s, %s, %s, %s)
            """, (firstname, lastname, gender, email, hashed_pw))
            conn.commit()
            user_id = cur.lastrowid
            cur.close()

            # Store session data including email
            session['user_id'] = user_id
            session['username'] = firstname
            session['user_email'] = email  # Store email in session

            flash("Account created successfully!", "success")
            return redirect(url_for('onboarding'))

        except mysql.connector.Error as err:
            flash(f"Database error: {err}", "error")
            return redirect(url_for('signup'))
        finally:
            conn.close()

    return render_template('signup.html')

# -------------------------
# Login
# -------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        if not conn:
            flash("Database connection error", "error")
            return redirect(url_for('login'))

        try:
            cur = conn.cursor()
            cur.execute("SELECT id, password, firstname, email FROM users WHERE email=%s", (email,))
            user = cur.fetchone()
            cur.close()
        except mysql.connector.Error as err:
            return f"Database error: {err}"
        finally:
            conn.close()

        if user:
            user_id, hashed_pw, firstname, user_email = user
            if check_password_hash(hashed_pw, password):
                session['user_id'] = user_id
                session['username'] = firstname
                session['user_email'] = user_email  # Store email in session
                
                # Check if user has completed onboarding
                conn = get_db_connection()
                if conn:
                    try:
                        cur = conn.cursor()
                        cur.execute("SELECT id FROM user_onboarding WHERE user_id = %s", (user_id,))
                        has_onboarding_data = cur.fetchone()
                        cur.close()
                        
                        if has_onboarding_data:
                            # User has completed onboarding - go to dashboard
                            return redirect(url_for('dashboard'))
                        else:
                            # User hasn't completed onboarding - go to onboarding
                            return redirect(url_for('onboarding'))
                            
                    except mysql.connector.Error as err:
                        print(f"Error checking onboarding status: {err}")
                        return redirect(url_for('onboarding'))
                    finally:
                        conn.close()
                else:
                    return redirect(url_for('onboarding'))
                    
            else:
                flash("Invalid email or password.", "error")
                return redirect(url_for('login'))
        else:
            flash("Email not registered.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')

# -------------------------
# Forgot Password
# -------------------------
@app.route('/forgotpassword')
def forgot_password():
    return render_template('forgotpassword.html')

# -------------------------
# Settings Route
# -------------------------
@app.route('/settings')
def settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_data = {}
    
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            
            # Get user data including phone number
            cur.execute("SELECT firstname, email, phone_number FROM users WHERE id = %s", (user_id,))
            user = cur.fetchone()
            cur.close()
            
            if user:
                user_data = {
                    'name': user['firstname'],
                    'email': user['email'],
                    'phone': user['phone_number'] or ''
                }
                
        except Exception as e:
            print(f"Error loading user data: {e}")
        finally:
            conn.close()
    
    return render_template('settings.html', 
                         user_name=session.get('username'),
                         user_email=session.get('user_email'),
                         user_data=user_data)

# -------------------------
# Update Profile Route
# -------------------------
@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    fullname = request.form.get('fullname')
    phone_number = request.form.get('phone_number')
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor()
        
        # Update user profile with phone number
        cur.execute("""
            UPDATE users 
            SET firstname = %s, phone_number = %s 
            WHERE id = %s
        """, (fullname, phone_number, user_id))
        
        conn.commit()
        cur.close()
        
        # Update session with new name
        session['username'] = fullname
        
        return jsonify({"success": True, "message": "Profile updated successfully!"})
        
    except Exception as e:
        print(f"Error updating profile: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error updating profile"}), 500
    finally:
        conn.close()

# -------------------------
# Change Password Route
# -------------------------
@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor()
        
        # Get current password hash
        cur.execute("SELECT password FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        
        if user and check_password_hash(user[0], current_password):
            # Update password
            hashed_new_password = generate_password_hash(new_password)
            cur.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_new_password, user_id))
            conn.commit()
            cur.close()
            
            return jsonify({"success": True, "message": "Password changed successfully!"})
        else:
            cur.close()
            return jsonify({"success": False, "message": "Current password is incorrect"}), 400
            
    except Exception as e:
        print(f"Error changing password: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error changing password"}), 500
    finally:
        conn.close()

# -------------------------
# Calendar Route
# -------------------------
@app.route('/calendar')
def calendar():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('calendar.html', user_name=session.get('username'))

# -------------------------
# Calendar Notes Routes
# -------------------------

@app.route('/get_calendar_notes', methods=['GET'])
def get_calendar_notes():
    """Get all notes for the logged-in user"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor(dictionary=True)
        
        cur.execute("""
            SELECT note_date, note_content, created_at 
            FROM calendar_notes 
            WHERE user_id = %s 
            ORDER BY note_date DESC, created_at DESC
        """, (user_id,))
        
        notes = cur.fetchall()
        cur.close()
        
        # Format the notes for the frontend
        notes_dict = {}
        for note in notes:
            date_str = note['note_date'].strftime('%Y-%m-%d')
            if date_str not in notes_dict:
                notes_dict[date_str] = []
            
            notes_dict[date_str].append({
                'content': note['note_content'],
                'timestamp': note['created_at'].isoformat()
            })
        
        return jsonify({"success": True, "notes": notes_dict})
        
    except Exception as e:
        print(f"Error fetching calendar notes: {e}")
        return jsonify({"success": False, "message": "Error fetching notes"}), 500
    finally:
        conn.close()

@app.route('/save_calendar_note', methods=['POST'])
def save_calendar_note():
    """Save a new calendar note"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    data = request.get_json()
    
    if not data or 'date' not in data or 'content' not in data:
        return jsonify({"success": False, "message": "Missing required fields"}), 400
    
    note_date = data['date']
    note_content = data['content'].strip()
    
    if not note_content:
        return jsonify({"success": False, "message": "Note content cannot be empty"}), 400
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor()
        
        # Insert the note
        cur.execute("""
            INSERT INTO calendar_notes (user_id, note_date, note_content)
            VALUES (%s, %s, %s)
        """, (user_id, note_date, note_content))
        
        conn.commit()
        note_id = cur.lastrowid
        
        # Get the created timestamp
        cur.execute("SELECT created_at FROM calendar_notes WHERE id = %s", (note_id,))
        created_at = cur.fetchone()[0]
        
        cur.close()
        
        return jsonify({
            "success": True, 
            "message": "Note saved successfully!",
            "timestamp": created_at.isoformat()
        })
        
    except Exception as e:
        print(f"Error saving calendar note: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error saving note"}), 500
    finally:
        conn.close()

@app.route('/delete_calendar_notes', methods=['POST'])
def delete_calendar_notes():
    """Delete all notes for a specific date"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    data = request.get_json()
    
    if not data or 'date' not in data:
        return jsonify({"success": False, "message": "Missing date parameter"}), 400
    
    note_date = data['date']
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor()
        
        cur.execute("""
            DELETE FROM calendar_notes 
            WHERE user_id = %s AND note_date = %s
        """, (user_id, note_date))
        
        conn.commit()
        deleted_count = cur.rowcount
        cur.close()
        
        return jsonify({
            "success": True, 
            "message": f"Deleted {deleted_count} notes",
            "deleted_count": deleted_count
        })
        
    except Exception as e:
        print(f"Error deleting calendar notes: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error deleting notes"}), 500
    finally:
        conn.close()

# -------------------------
# Blog Routes
# -------------------------

@app.route('/get_blog/<int:blog_id>')
def get_blog(blog_id):
    """Get single blog for modal"""
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor(dictionary=True)
        
        cur.execute("SELECT * FROM blogs WHERE id = %s AND status = 'published'", (blog_id,))
        blog = cur.fetchone()
        
        cur.close()
        
        if blog:
            return jsonify({
                "success": True,
                "blog": {
                    "id": blog['id'],
                    "title": blog['title'],
                    "content": blog['content'],
                    "category": blog['category'],
                    "category_color": blog['category_color'],
                    "author": blog['author'],
                    "read_time": blog['read_time'],
                    "image_url": blog['image_url'],
                    "created_at": blog['created_at'].strftime('%b %d, %Y') if blog['created_at'] else ''
                }
            })
        else:
            return jsonify({"success": False, "message": "Blog not found"}), 404
            
    except Exception as e:
        print(f"Error fetching blog: {e}")
        return jsonify({"success": False, "message": "Error fetching blog"}), 500
    finally:
        conn.close()

@app.route('/admin/blogs')
def admin_blogs():
    """Admin panel for managing blogs"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if user is admin (you might want to add an admin field to your users table)
    # For now, we'll allow any logged-in user to access
    return render_template('admin_blogs.html', user_name=session.get('username'))

@app.route('/admin/create_blog', methods=['POST'])
def create_blog():
    """Create a new blog"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        data = request.get_json()
        
        title = data.get('title')
        content = data.get('content')
        excerpt = data.get('excerpt', '')
        category = data.get('category')
        category_color = data.get('category_color', '#abc85f')
        author = data.get('author')
        read_time = data.get('read_time')
        image_url = data.get('image_url', '')
        is_featured = data.get('is_featured', False)
        status = data.get('status', 'published')
        
        # Validate required fields
        if not all([title, content, category, author, read_time]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400
        
        cur = conn.cursor()
        
        # If setting as featured, unfeature other blogs
        if is_featured:
            cur.execute("UPDATE blogs SET is_featured = FALSE WHERE is_featured = TRUE")
        
        # Insert new blog
        cur.execute("""
            INSERT INTO blogs (title, content, excerpt, category, category_color, author, read_time, image_url, is_featured, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (title, content, excerpt, category, category_color, author, read_time, image_url, is_featured, status))
        
        conn.commit()
        blog_id = cur.lastrowid
        
        cur.close()
        
        return jsonify({
            "success": True, 
            "message": "Blog created successfully!",
            "blog_id": blog_id
        })
        
    except Exception as e:
        print(f"Error creating blog: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error creating blog"}), 500
    finally:
        conn.close()

@app.route('/admin/get_blogs')
def admin_get_blogs():
    """Get all blogs for admin panel"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor(dictionary=True)
        
        cur.execute("SELECT * FROM blogs ORDER BY created_at DESC")
        blogs = cur.fetchall()
        
        cur.close()
        
        # Convert datetime objects to strings
        for blog in blogs:
            if blog['created_at']:
                blog['created_at'] = blog['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if blog['updated_at']:
                blog['updated_at'] = blog['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({"success": True, "blogs": blogs})
        
    except Exception as e:
        print(f"Error fetching blogs: {e}")
        return jsonify({"success": False, "message": "Error fetching blogs"}), 500
    finally:
        conn.close()

@app.route('/admin/update_blog/<int:blog_id>', methods=['POST'])
def update_blog(blog_id):
    """Update a blog"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        data = request.get_json()
        
        cur = conn.cursor()
        
        # Build dynamic update query
        update_fields = []
        update_values = []
        
        if 'title' in data:
            update_fields.append("title = %s")
            update_values.append(data['title'])
        if 'content' in data:
            update_fields.append("content = %s")
            update_values.append(data['content'])
        if 'excerpt' in data:
            update_fields.append("excerpt = %s")
            update_values.append(data['excerpt'])
        if 'category' in data:
            update_fields.append("category = %s")
            update_values.append(data['category'])
        if 'category_color' in data:
            update_fields.append("category_color = %s")
            update_values.append(data['category_color'])
        if 'author' in data:
            update_fields.append("author = %s")
            update_values.append(data['author'])
        if 'read_time' in data:
            update_fields.append("read_time = %s")
            update_values.append(data['read_time'])
        if 'image_url' in data:
            update_fields.append("image_url = %s")
            update_values.append(data['image_url'])
        if 'is_featured' in data:
            update_fields.append("is_featured = %s")
            update_values.append(data['is_featured'])
            # If setting as featured, unfeature other blogs
            if data['is_featured']:
                cur.execute("UPDATE blogs SET is_featured = FALSE WHERE id != %s", (blog_id,))
        if 'status' in data:
            update_fields.append("status = %s")
            update_values.append(data['status'])
        
        if update_fields:
            update_values.append(blog_id)
            query = f"UPDATE blogs SET {', '.join(update_fields)} WHERE id = %s"
            cur.execute(query, update_values)
            conn.commit()
        
        cur.close()
        
        return jsonify({"success": True, "message": "Blog updated successfully!"})
        
    except Exception as e:
        print(f"Error updating blog: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error updating blog"}), 500
    finally:
        conn.close()

@app.route('/admin/delete_blog/<int:blog_id>', methods=['DELETE'])
def delete_blog(blog_id):
    """Delete a blog"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "Database connection error"}), 500
    
    try:
        cur = conn.cursor()
        
        cur.execute("DELETE FROM blogs WHERE id = %s", (blog_id,))
        conn.commit()
        
        cur.close()
        
        return jsonify({"success": True, "message": "Blog deleted successfully!"})
        
    except Exception as e:
        print(f"Error deleting blog: {e}")
        conn.rollback()
        return jsonify({"success": False, "message": "Error deleting blog"}), 500
    finally:
        conn.close()

# -------------------------
# Menstrual Cycle Tracking
# -------------------------
@app.route('/cycle')
def cycle_form():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if user has existing cycle data
    user_id = session['user_id']
    existing_data = None
    
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT last_period, cycle_length, period_duration 
                FROM menstrual_cycle 
                WHERE user_id = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (user_id,))
            existing_data = cur.fetchone()
            cur.close()
        except mysql.connector.Error as err:
            print(f"DEBUG: Error fetching existing cycle data: {err}")
        finally:
            conn.close()
    
    # Pass current date to template for max date validation
    current_date = datetime.now().strftime("%Y-%m-%d")
    return render_template('cycle.html', existing_data=existing_data, current_date=current_date)

@app.route('/save_cycle', methods=['POST'])
def save_cycle():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    last_period_str = request.form['lastPeriod']
    cycle_length = int(request.form['cycleLength'])
    period_duration = int(request.form['periodDuration'])

    print(f"DEBUG: Received form data - user_id: {user_id}, last_period: {last_period_str}, cycle_length: {cycle_length}, period_duration: {period_duration}")

    # Validate inputs
    if cycle_length < 21 or cycle_length > 35:
        flash("Cycle length should typically be between 21-35 days.", "error")
        return redirect(url_for('cycle_form'))
    
    if period_duration < 2 or period_duration > 7:
        flash("Period duration should typically be between 2-7 days.", "error")
        return redirect(url_for('cycle_form'))

    # Validate date is not in future
    last_period = datetime.strptime(last_period_str, "%Y-%m-%d")
    if last_period > datetime.now():
        flash("Last period date cannot be in the future.", "error")
        return redirect(url_for('cycle_form'))

    # Get user age for ML features
    user_age = None
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT age FROM user_onboarding WHERE user_id = %s", (user_id,))
            result = cur.fetchone()
            user_age = result[0] if result else None
            cur.close()
        except:
            pass
        finally:
            conn.close()

    # PREDICT USING ML
    predictions = predict_next_cycle_ml(
        user_id=user_id,
        last_period=last_period,
        cycle_length=cycle_length,
        period_duration=period_duration,
        age=user_age
    )

    conn = get_db_connection()
    if not conn:
        flash("Database connection error", "error")
        return redirect(url_for('cycle_form'))
    
    try:
        cur = conn.cursor()
        
        # Insert data using only the columns that exist in your table
        cur.execute("""
            INSERT INTO menstrual_cycle 
            (user_id, last_period, cycle_length, period_duration)
            VALUES (%s, %s, %s, %s)
        """, (user_id, last_period_str, cycle_length, period_duration))
        
        conn.commit()
        print("DEBUG: Cycle data saved successfully!")
        flash("Cycle data saved successfully!", "success")
        
    except mysql.connector.Error as err:
        print(f"DEBUG: MySQL Error: {err}")
        flash(f"Database error: {err.msg}", "error")
        return redirect(url_for('cycle_form'))
    finally:
        if 'cur' in locals():
            cur.close()
        conn.close()

    fertile_start, fertile_end = predictions['fertile_window']
    
    return render_template('results.html',
                           next_period=predictions['next_period'].strftime("%B %d, %Y"),
                           ovulation_day=predictions['ovulation_day'].strftime("%B %d, %Y"),
                           fertile_window=f"{fertile_start.strftime('%B %d, %Y')} - {fertile_end.strftime('%B %d, %Y')}",
                           prediction_method=predictions['method'].upper(),
                           predicted_cycle_length=predictions['predicted_cycle_length'])

# -------------------------
# Cycle History Route
# -------------------------
@app.route('/cycle_history')
def cycle_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    cycle_history = []
    
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT last_period, cycle_length, period_duration, created_at
                FROM menstrual_cycle 
                WHERE user_id = %s 
                ORDER BY created_at DESC
            """, (user_id,))
            cycle_history = cur.fetchall()
            cur.close()
        except mysql.connector.Error as err:
            print(f"DEBUG: Error retrieving cycle history: {err}")
            flash("Error retrieving cycle history.", "error")
        finally:
            conn.close()
    
    return render_template('cycle_history.html', cycle_history=cycle_history, user_name=session.get('username'))

# -------------------------
# ML Training Route
# -------------------------
@app.route('/train_model')
def train_model_route():
    """Route to train ML model"""
    if train_ml_model():
        return "ML model trained successfully!"
    else:
        return "Error training ML model!"

# -------------------------
# Logout
# -------------------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('user_email', None)
    return redirect(url_for('login'))

# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)