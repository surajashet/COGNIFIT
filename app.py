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

# MySQL database configuration
db_cognifit = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "cognifit",
    "port": 3306
}



# -------------------------
# Database Initialization
# -------------------------
def init_database():
    """Initialize database with required columns and tables"""
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor()
        
        # Create users table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                firstname VARCHAR(100) NOT NULL,
                lastname VARCHAR(100) NOT NULL,
                gender VARCHAR(10) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                phone_number VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("Ensured users table exists")

        # Create user_onboarding table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_onboarding (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                goals TEXT,
                activity_level VARCHAR(50),
                age INT,
                height DECIMAL(5,2),
                weight DECIMAL(5,2),
                injury_conditions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        print("Ensured user_onboarding table exists")
        
        # Check if phone_number column exists in users table
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = 'cognifit_db' 
            AND TABLE_NAME = 'users' 
            AND COLUMN_NAME = 'phone_number'
        """)
        column_exists = cur.fetchone()[0]
        
        if not column_exists:
            # Add phone_number column if it doesn't exist
            cur.execute("ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)")
            print("Added phone_number column to users table")
        
        # Check if menstrual_cycle table exists and has proper structure
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = 'cognifit_db' 
            AND TABLE_NAME = 'menstrual_cycle'
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            # Create menstrual_cycle table with user_id
            cur.execute("""
                CREATE TABLE menstrual_cycle (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    last_period DATE NOT NULL,
                    cycle_length INT NOT NULL,
                    period_duration INT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user (user_id)
                )
            """)
            print("Created menstrual_cycle table")
        
        # Check if calendar_notes table exists
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = 'cognifit_db' 
            AND TABLE_NAME = 'calendar_notes'
        """)
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
            WHERE TABLE_SCHEMA = 'cognifit_db' 
            AND TABLE_NAME = 'blogs'
        """)
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
            WHERE TABLE_SCHEMA = 'cognifit_db' 
            AND TABLE_NAME = 'recipes'
        """)
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
        
        # Check if workouts table exists
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = 'cognifit_db' 
            AND TABLE_NAME = 'workouts'
        """)
        workouts_table_exists = cur.fetchone()[0]
        
        if not workouts_table_exists:
            # Create workouts table
            cur.execute("""
                CREATE TABLE workouts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    workout_type VARCHAR(100) NOT NULL,
                    workout_date DATE NOT NULL,
                    duration_minutes INT NOT NULL,
                    intensity_level ENUM('light', 'moderate', 'vigorous') NOT NULL,
                    calories_burned INT NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user_date (user_id, workout_date),
                    INDEX idx_type (workout_type)
                )
            """)
            print("Created workouts table")
        
        # Check if nutrition_logs table exists
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = 'cognifit_db' 
            AND TABLE_NAME = 'nutrition_logs'
        """)
        nutrition_table_exists = cur.fetchone()[0]
        
        if not nutrition_table_exists:
            # Create nutrition_logs table
            cur.execute("""
                CREATE TABLE nutrition_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    log_date DATE NOT NULL,
                    calories INT,
                    protein_g INT,
                    carbs_g INT,
                    fat_g INT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user_date (user_id, log_date)
                )
            """)
            print("Created nutrition_logs table")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error initializing database: {e}")

# Initialize database when app starts
init_database()

def predict_next_cycle_rule_based(user_id, last_period, cycle_length, period_duration, age=None):
    """Rule-based prediction using historical data and realistic patterns"""
    try:
        # Get user's historical data - FILTERED BY USER_ID
        historical_data = get_user_cycle_history(user_id)
        
        if historical_data and len(historical_data) >= 2:
            # Use historical data for more accurate prediction
            cycle_lengths = [entry['cycle_length'] for entry in historical_data]
            
            # Ensure realistic cycle lengths and calculate weighted average
            realistic_lengths = [cl for cl in cycle_lengths if 21 <= cl <= 35]
            
            if realistic_lengths:
                # Give more weight to recent cycles
                recent_weight = 0.6  # 60% weight to most recent
                historical_weight = 0.4  # 40% weight to historical average
                
                if len(realistic_lengths) >= 3:
                    # Use weighted average of recent cycles
                    recent_avg = np.mean(realistic_lengths[-2:])  # Last 2 cycles
                    historical_avg = np.mean(realistic_lengths[:-2])  # Older cycles
                    predicted_cycle_length = (recent_avg * recent_weight + 
                                            historical_avg * historical_weight)
                else:
                    # Simple average for fewer cycles
                    predicted_cycle_length = np.mean(realistic_lengths)
                
                # Add small random variation (1-2 days) to simulate natural variation
                variation = np.random.uniform(-1.5, 1.5)
                predicted_cycle_length += variation
            else:
                # Fallback to current cycle length with slight variation
                predicted_cycle_length = cycle_length + np.random.uniform(-1, 1)
        else:
            # For new users or insufficient data, use current cycle length
            predicted_cycle_length = cycle_length
        
        # Ensure realistic prediction range
        predicted_cycle_length = max(21, min(35, predicted_cycle_length))
        
        # Age-based adjustments
        if age:
            if age < 20:
                # Teenagers often have less regular cycles
                predicted_cycle_length += np.random.uniform(-2, 2)
            elif age > 35:
                # Women in late 30s/40s might have shorter cycles
                predicted_cycle_length -= np.random.uniform(0, 1)
        
        # Final range check
        predicted_cycle_length = max(21, min(35, predicted_cycle_length))
        
        # Calculate dates
        next_period = last_period + timedelta(days=int(predicted_cycle_length))
        ovulation_day = last_period + timedelta(days=(int(predicted_cycle_length) - 14))
        fertile_start = ovulation_day - timedelta(days=4)
        fertile_end = ovulation_day + timedelta(days=1)
        
        return {
            'next_period': next_period,
            'ovulation_day': ovulation_day,
            'fertile_window': (fertile_start, fertile_end),
            'predicted_cycle_length': int(predicted_cycle_length),
            'method': 'rule_based_historical' if historical_data and len(historical_data) >= 2 else 'rule_based_default'
        }
        
    except Exception as e:
        print(f"Rule-based prediction failed: {e}")
        # Fallback to traditional calculation
        return predict_traditional(last_period, cycle_length)

def predict_traditional(last_period, cycle_length):
    """Traditional prediction method with realistic cycle lengths"""
    # Ensure cycle length is realistic
    realistic_cycle_length = max(21, min(35, cycle_length))
    
    next_period = last_period + timedelta(days=realistic_cycle_length)
    ovulation_day = last_period + timedelta(days=(realistic_cycle_length - 14))
    fertile_start = ovulation_day - timedelta(days=4)
    fertile_end = ovulation_day + timedelta(days=1)
    
    return {
        'next_period': next_period,
        'ovulation_day': ovulation_day,
        'fertile_window': (fertile_start, fertile_end),
        'predicted_cycle_length': realistic_cycle_length,
        'method': 'traditional'
    }

def get_user_cycle_history(user_id):
    """Get user's historical cycle data - FILTERED BY USER_ID"""
    try:
        conn = mysql.connector.connect(**db_cognifit)
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
        conn.close()
        return history
    except Exception as e:
        print(f"Error getting user history: {e}")
        return []
def get_user_weight(user_id):
    """Get user's weight from onboarding data"""
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor()
        
        cur.execute("SELECT weight FROM user_onboarding WHERE user_id = %s", (user_id,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        return result[0] if result else None
        
    except Exception as e:
        print(f"Error fetching user weight: {e}")
        return None
    
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
# Workout Helper Functions
# -------------------------

def calculate_calories_burned(user_id,workout_type, duration_minutes, intensity_level):
    """Calculate calories burned based on workout type, duration, and intensity"""
    # MET values for different workout types (calories burned per kg per hour)
    MET_VALUES = {
        'running': 9.8,
        'cycling': 7.5,
        'swimming': 8.0,
        'weight-training': 6.0,
        'yoga': 3.0,
        'hiit': 8.5,
        'walking': 4.0
    }
    
    # Intensity multipliers
    INTENSITY_MULTIPLIERS = {
        'light': 0.8,
        'moderate': 1.0,
        'vigorous': 1.3
    }
    
    # Assume average weight of 70kg for calculation
    # Get user's actual weight from database
    user_weight = get_user_weight(user_id)
    
    # If weight not available, fallback to 70kg
    if not user_weight:
        user_weight = 70
    
    weight_kg = user_weight

    met_value = MET_VALUES.get(workout_type, 5.0)
    intensity_multiplier = INTENSITY_MULTIPLIERS.get(intensity_level, 1.0)
    
    # Calories = MET * weight(kg) * time(hours) * intensity multiplier
    calories = met_value * float(weight_kg) * (float(duration_minutes) / 60) * intensity_multiplier
    return int(calories)

def get_workout_stats(user_id):
    """Get workout statistics for a user"""
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
        
        # Get total workouts, calories, and minutes
        cur.execute("""
            SELECT 
                COUNT(*) as total_workouts,
                SUM(calories_burned) as total_calories,
                SUM(duration_minutes) as total_minutes
            FROM workouts 
            WHERE user_id = %s
        """, (user_id,))
        stats = cur.fetchone()
        
        # Get workout distribution by type
        cur.execute("""
            SELECT 
                workout_type,
                SUM(duration_minutes) as total_minutes
            FROM workouts 
            WHERE user_id = %s
            GROUP BY workout_type
        """, (user_id,))
        workout_distribution = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {
            'total_workouts': stats['total_workouts'] or 0,
            'total_calories': stats['total_calories'] or 0,
            'total_minutes': stats['total_minutes'] or 0,
            'workout_distribution': workout_distribution
        }
        
    except Exception as e:
        print(f"Error getting workout stats: {e}")
        return {
            'total_workouts': 0,
            'total_calories': 0,
            'total_minutes': 0,
            'workout_distribution': []
        }

# -------------------------
# Progress Helper Functions - UPDATED
# -------------------------

def calculate_progress_metrics(workout_data, time_range='7'):
    """Calculate comprehensive progress metrics for the progress page"""
    today = datetime.now().date()
    
    # Calculate date range based on filter
    if time_range == '7':
        start_date = today - timedelta(days=7)
        period_label = 'week'
    elif time_range == '30':
        start_date = today - timedelta(days=30)
        period_label = 'month'
    elif time_range == '90':
        start_date = today - timedelta(days=90)
        period_label = '3 months'
    elif time_range == '365':
        start_date = today - timedelta(days=365)
        period_label = 'year'
    else:  # 'all'
        start_date = None
        period_label = 'all time'
    
    # Filter workouts by date range if specified
    if start_date:
        filtered_workouts = [w for w in workout_data if w['workout_date'] >= start_date]
    else:
        filtered_workouts = workout_data
    
    # Calculate basic metrics
    total_workouts = len(filtered_workouts)
    total_calories = sum(w['calories_burned'] for w in filtered_workouts)
    total_minutes = sum(w['duration_minutes'] for w in filtered_workouts)
    
    # Calculate averages
    avg_calories_per_workout = total_calories / total_workouts if total_workouts > 0 else 0
    avg_duration_per_workout = total_minutes / total_workouts if total_workouts > 0 else 0
    
    # Calculate consistency (workout days vs total days in period)
    if start_date:
        total_days = (today - start_date).days
        workout_days = len(set(w['workout_date'] for w in filtered_workouts))
        consistency = (workout_days / total_days) * 100 if total_days > 0 else 0
    else:
        consistency = 0
    
    return {
        'total_workouts': total_workouts,
        'total_calories': total_calories,
        'total_minutes': total_minutes,
        'avg_calories_per_workout': avg_calories_per_workout,
        'avg_duration_per_workout': avg_duration_per_workout,
        'consistency': consistency,
        'period_label': period_label
    }

def get_workout_frequency_data(workout_data, time_range='7'):
    """Get workout frequency data for charts"""
    today = datetime.now().date()
    
    # Calculate date range
    if time_range == '7':
        days = 7
        period_type = 'daily'
    elif time_range == '30':
        days = 30
        period_type = 'daily'
    elif time_range == '90':
        days = 90
        period_type = 'weekly'
    elif time_range == '365':
        days = 365
        period_type = 'monthly'
    else:  # 'all'
        # For all time, group by month
        period_type = 'monthly'
        days = None
    
    if period_type == 'daily':
        # Group by day for last X days
        frequency_data = []
        for i in range(days):
            date = today - timedelta(days=i)
            day_workouts = [w for w in workout_data if w['workout_date'] == date]
            frequency_data.append({
                'period': date.strftime('%Y-%m-%d'),
                'workout_count': len(day_workouts),
                'total_calories': sum(w['calories_burned'] for w in day_workouts),
                'total_minutes': sum(w['duration_minutes'] for w in day_workouts)
            })
        frequency_data.reverse()  # Reverse to show chronological order
    
    elif period_type == 'weekly':
        # Group by week for last 12 weeks
        frequency_data = []
        for i in range(12):
            week_end = today - timedelta(weeks=i)
            week_start = week_end - timedelta(days=6)
            week_workouts = [w for w in workout_data if week_start <= w['workout_date'] <= week_end]
            frequency_data.append({
                'period': f"Week {12-i}",
                'workout_count': len(week_workouts),
                'total_calories': sum(w['calories_burned'] for w in week_workouts),
                'total_minutes': sum(w['duration_minutes'] for w in week_workouts)
            })
        frequency_data.reverse()
    
    else:  # monthly
        # Group by month for last 12 months
        frequency_data = []
        for i in range(12):
            month_date = today.replace(day=1) - timedelta(days=30*i)
            month_start = month_date.replace(day=1)
            if month_start.month == 12:
                month_end = month_start.replace(year=month_start.year+1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = month_start.replace(month=month_start.month+1, day=1) - timedelta(days=1)
            
            month_workouts = [w for w in workout_data if month_start <= w['workout_date'] <= month_end]
            frequency_data.append({
                'period': month_start.strftime('%b %Y'),
                'workout_count': len(month_workouts),
                'total_calories': sum(w['calories_burned'] for w in month_workouts),
                'total_minutes': sum(w['duration_minutes'] for w in month_workouts)
            })
        frequency_data.reverse()
    
    return frequency_data

# -------------------------
# Routes
# -------------------------

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/snap')
def snap():
    return render_template('snap.html')


# -------------------------
# Progress Routes - UPDATED: Proper workout data aggregation
# -------------------------

@app.route('/progress')
def progress():
    """Main progress page route"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('progress.html', 
                         user_name=session.get('username'))
from flask import request, jsonify
import mysql.connector


@app.route('/api/food/progress/<int:user_id>')
def food_progress(user_id):
    conn = None
    try:
        range_param = request.args.get('range', '7')  # can be 7, 30, 365, or 'all'

        # Determine the range filter
        if range_param == 'all':
            range_days = None
        else:
            try:
                range_days = int(range_param)
            except ValueError:
                range_days = 7  # fallback just in case

        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='cognifit'
        )
        cursor = conn.cursor(dictionary=True)

        if range_days:  # For specific range (7, 30, 365, etc.)
            start_date = datetime.now() - timedelta(days=range_days)
            query = """
                SELECT DATE(log_date) AS log_date,
                       SUM(calories) AS total_calories,
                       SUM(protein) AS total_protein,
                       SUM(carbs) AS total_carbs,
                       SUM(fat) AS total_fat
                FROM food_log
                WHERE user_id = %s AND log_date >= %s
                GROUP BY DATE(log_date)
                ORDER BY DATE(log_date)
            """
            cursor.execute(query, (user_id, start_date))
        else:  # For all time data
            query = """
                SELECT DATE(log_date) AS log_date,
                       SUM(calories) AS total_calories,
                       SUM(protein) AS total_protein,
                       SUM(carbs) AS total_carbs,
                       SUM(fat) AS total_fat
                FROM food_log
                WHERE user_id = %s
                GROUP BY DATE(log_date)
                ORDER BY DATE(log_date)
            """
            cursor.execute(query, (user_id,))

        data = cursor.fetchall()
        return jsonify({"success": True, "data": data})

    except Exception as e:
        print("Error in /api/food/progress:", e)
        return jsonify({"success": False, "message": str(e)})

    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


# -------------------------
# NEW API ROUTE FOR CYCLE PREDICTION
# -------------------------

@app.route('/api/predict_cycle', methods=['POST'])
def api_predict_cycle():
    """API endpoint for cycle prediction using RULE-BASED method"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    data = request.get_json()
    
    if not data or 'dates' not in data:
        return jsonify({"success": False, "message": "No dates provided"}), 400
    
    dates = data.get('dates', [])
    
    if len(dates) < 1:
        return jsonify({"success": False, "message": "Need at least one date"}), 400
    
    try:
        # Sort dates and get the most recent period date
        sorted_dates = sorted(dates)
        last_period_str = sorted_dates[-1]
        last_period = datetime.strptime(last_period_str, "%Y-%m-%d")
        
        # Calculate cycle lengths from provided dates
        cycle_lengths = []
        for i in range(len(sorted_dates) - 1):
            start = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
            end = datetime.strptime(sorted_dates[i + 1], "%Y-%m-%d")
            diff_days = (end - start).days
            cycle_lengths.append(diff_days)
        
        # Use average of realistic cycle lengths, or default
        realistic_lengths = [cl for cl in cycle_lengths if 21 <= cl <= 35]
        if realistic_lengths:
            avg_cycle_length = sum(realistic_lengths) / len(realistic_lengths)
        else:
            avg_cycle_length = 28  # Default realistic cycle length
        
        # Get user age for rule-based adjustments
        user_age = None
        try:
            conn = mysql.connector.connect(**db_cognifit)
            cur = conn.cursor()
            cur.execute("SELECT age FROM user_onboarding WHERE user_id = %s", (user_id,))
            result = cur.fetchone()
            user_age = result[0] if result else None
            cur.close()
            conn.close()
        except:
            pass
        
        # Use RULE-BASED prediction (replaced ML)
        prediction = predict_next_cycle_rule_based(
            user_id=user_id,
            last_period=last_period,
            cycle_length=int(avg_cycle_length),
            period_duration=5,  # Default period duration
            age=user_age
        )
        
        # Save this prediction to database for future reference
        try:
            conn = mysql.connector.connect(**db_cognifit)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO menstrual_cycle 
                (user_id, last_period, cycle_length, period_duration)
                VALUES (%s, %s, %s, %s)
            """, (user_id, last_period_str, int(avg_cycle_length), 5))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error saving cycle data: {e}")
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "cycle_lengths": cycle_lengths,
            "avg_cycle_length": avg_cycle_length
        })
        
    except Exception as e:
        print(f"Error in cycle prediction: {e}")
        return jsonify({"success": False, "message": "Error processing prediction"}), 500

# -------------------------
# Recipe Routes
# -------------------------

# -------------------------
# Recipe Routes
# -------------------------
 
# -------------------------
# Recipe Routes
# -------------------------
 
@app.route('/recipes')
def recipes():
    if 'user_id' not in session:
        return redirect(url_for('login'))
   
    try:
        conn = mysql.connector.connect(**db_cognifit)
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
        conn.close()
       
        return render_template('recipes.html',
                             user_name=session.get('username'),
                             featured_recipe=featured_recipe,
                             recipes=recipes)
       
    except Exception as e:
        print(f"Error loading recipes: {e}")
        # Fallback to empty data
        return render_template('recipes.html',
                             user_name=session.get('username'),
                             featured_recipe=None,
                             recipes=[])
 
@app.route('/get_recipe/<int:recipe_id>')
def get_recipe(recipe_id):
    """Get single recipe for modal"""
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
       
        cur.execute("SELECT * FROM recipes WHERE id = %s", (recipe_id,))
        recipe = cur.fetchone()
       
        cur.close()
        conn.close()
       
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
                    "user_id": recipe['user_id'],  # ADD THIS LINE
                    "created_at": recipe['created_at'].strftime('%b %d, %Y') if recipe['created_at'] else ''
                }
            })
        else:
            return jsonify({"success": False, "message": "Recipe not found"}), 404
           
    except Exception as e:
        print(f"Error fetching recipe: {e}")
        return jsonify({"success": False, "message": "Error fetching recipe"}), 500
 
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
        user_id = session['user_id']  # ADD THIS LINE
       
        # Validate required fields
        if not all([title, category, total_time, calories, servings, ingredients, instructions]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400
       
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor()
       
        # If setting as featured, unfeature other recipes
        if is_featured:
            cur.execute("UPDATE recipes SET is_featured = FALSE WHERE is_featured = TRUE")
       
        # Insert new recipe WITH user_id
        cur.execute("""
            INSERT INTO recipes (title, category, prep_time, cook_time, total_time, difficulty, calories, servings, tags, image_url, ingredients, instructions, is_featured, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (title, category, prep_time, cook_time, total_time, difficulty, calories, servings, tags, image_url, ingredients, instructions, is_featured, user_id))
       
        conn.commit()
        recipe_id = cur.lastrowid
       
        cur.close()
        conn.close()
       
        return jsonify({
            "success": True,
            "message": "Recipe created successfully!",
            "recipe_id": recipe_id
        })
       
    except Exception as e:
        print(f"Error creating recipe: {e}")
        return jsonify({"success": False, "message": "Error creating recipe"}), 500
   
@app.route('/admin/get_recipes')
def admin_get_recipes():
    """Get all recipes for admin panel"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
   
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
       
        # Make sure this query includes user_id
        cur.execute("SELECT *, COALESCE(user_id, 0) as user_id FROM recipes ORDER BY created_at DESC")
        recipes = cur.fetchall()
       
        cur.close()
        conn.close()
       
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
   
@app.route('/admin/update_recipe/<int:recipe_id>', methods=['POST'])
def update_recipe(recipe_id):
    """Update a recipe"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
   
    try:
        data = request.get_json()
        user_id = session['user_id']  # ADD THIS LINE
       
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)  # Change to dictionary=True
       
        # First, check if the current user owns this recipe
        cur.execute("SELECT user_id FROM recipes WHERE id = %s", (recipe_id,))
        recipe = cur.fetchone()
       
        if not recipe:
            return jsonify({"success": False, "message": "Recipe not found"}), 404
       
        # Convert both to integers for proper comparison
        recipe_user_id = int(recipe['user_id']) if recipe['user_id'] is not None else 0
        session_user_id = int(user_id)
       
        # Check ownership - only allow if user owns the recipe
        if recipe_user_id != session_user_id:
            return jsonify({"success": False, "message": "You can only edit your own recipes"}), 403
       
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
        conn.close()
       
        return jsonify({"success": True, "message": "Recipe updated successfully!"})
       
    except Exception as e:
        print(f"Error updating recipe: {e}")
        return jsonify({"success": False, "message": "Error updating recipe"}), 500
   
@app.route('/admin/delete_recipe/<int:recipe_id>', methods=['DELETE'])
def delete_recipe(recipe_id):
    """Delete a recipe"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
   
    try:
        user_id = session['user_id']  # ADD THIS LINE
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)  # Change to dictionary=True
       
        # First, check if the current user owns this recipe
        cur.execute("SELECT user_id FROM recipes WHERE id = %s", (recipe_id,))
        recipe = cur.fetchone()
       
        if not recipe:
            return jsonify({"success": False, "message": "Recipe not found"}), 404
       
        # Convert both to integers for proper comparison
        recipe_user_id = int(recipe['user_id']) if recipe['user_id'] is not None else 0
        session_user_id = int(user_id)
       
        # Check ownership - only allow if user owns the recipe
        if recipe_user_id != session_user_id:
            return jsonify({"success": False, "message": "You can only delete your own recipes"}), 403
       
        cur.execute("DELETE FROM recipes WHERE id = %s", (recipe_id,))
        conn.commit()
       
        cur.close()
        conn.close()
       
        return jsonify({"success": True, "message": "Recipe deleted successfully!"})
       
    except Exception as e:
        print(f"Error deleting recipe: {e}")
        return jsonify({"success": False, "message": "Error deleting recipe"}), 500

@app.route('/blogs')
def blogs():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        conn = mysql.connector.connect(**db_cognifit)
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
        conn.close()
        
        return render_template('blogs.html', 
                             user_name=session.get('username'),
                             featured_blog=featured_blog,
                             blogs=blogs)
        
    except Exception as e:
        print(f"Error loading blogs: {e}")
        # Fallback to empty data
        return render_template('blogs.html', 
                             user_name=session.get('username'),
                             featured_blog=None,
                             blogs=[])

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
    
    try:
        # Get user onboarding data including height and weight
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT age, height, weight, activity_level, goals 
            FROM user_onboarding 
            WHERE user_id = %s
        """, (user_id,))
        user_data = cur.fetchone()
        cur.close()
        conn.close()
        
        # Calculate BMI if height and weight are available
        if user_data and user_data.get('height') and user_data.get('weight'):
            bmi = calculate_bmi(user_data['weight'], user_data['height'])
            bmi_category = get_bmi_category(bmi)
            
    except mysql.connector.Error as err:
        print(f"Error fetching user data for dashboard: {err}")
        flash("Error loading your health data", "error")
    
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
            
            # Connect to database and save onboarding data
            conn = mysql.connector.connect(**db_cognifit)
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
            conn.close()
            
            return jsonify({"success": True, "message": "Onboarding data saved successfully!"})
            
        except Exception as e:
            print(f"Error saving onboarding data: {e}")
            return jsonify({"success": False, "message": "Error saving data"}), 500

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

        try:
            conn = mysql.connector.connect(**db_cognifit)
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
            conn.close()

            # Store session data including email
            session['user_id'] = user_id
            session['username'] = firstname
            session['user_email'] = email  # Store email in session

            flash("Account created successfully!", "success")
            return redirect(url_for('onboarding'))

        except mysql.connector.Error as err:
            flash(f"Database error: {err}", "error")
            return redirect(url_for('signup'))

    return render_template('signup.html')

# -------------------------
# Login - CORRECTED VERSION
# -------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        # Validate required fields
        if not email or not password:
            flash("Please fill in all fields.", "error")
            return redirect(url_for('login'))

        try:
            conn = mysql.connector.connect(**db_cognifit)
            cur = conn.cursor()
            cur.execute("SELECT id, password, firstname, email FROM users WHERE email=%s", (email,))
            user = cur.fetchone()
            cur.close()
            conn.close()
        except mysql.connector.Error as err:
            flash("Database error. Please try again.", "error")
            return redirect(url_for('login'))

        if user:
            user_id, hashed_pw, firstname, user_email = user
            if check_password_hash(hashed_pw, password):
                session['user_id'] = user_id
                session['username'] = firstname
                session['user_email'] = user_email
                
                # Check if user has completed onboarding
                try:
                    conn = mysql.connector.connect(**db_cognifit)
                    cur = conn.cursor()
                    cur.execute("SELECT id FROM user_onboarding WHERE user_id = %s", (user_id,))
                    has_onboarding_data = cur.fetchone()
                    cur.close()
                    conn.close()
                    
                    if has_onboarding_data:
                        return redirect(url_for('dashboard'))
                    else:
                        return redirect(url_for('onboarding'))
                        
                except mysql.connector.Error as err:
                    print(f"Error checking onboarding status: {err}")
                    return redirect(url_for('onboarding'))
                    
            else:
                flash("Invalid email or password.", "error")
                return redirect(url_for('login'))
        else:
            flash("Email not registered.", "error")
            return redirect(url_for('login'))

    # GET request - just render the template
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
    
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
        
        # Get user data including phone number
        cur.execute("SELECT firstname, email, phone_number FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        
        if user:
            user_data = {
                'name': user['firstname'],
                'email': user['email'],
                'phone': user['phone_number'] or ''
            }
            
    except Exception as e:
        print(f"Error loading user data: {e}")
    
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
    
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor()
        
        # Update user profile with phone number
        cur.execute("""
            UPDATE users 
            SET firstname = %s, phone_number = %s 
            WHERE id = %s
        """, (fullname, phone_number, user_id))
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Update session with new name
        session['username'] = fullname
        
        return jsonify({"success": True, "message": "Profile updated successfully!"})
        
    except Exception as e:
        print(f"Error updating profile: {e}")
        return jsonify({"success": False, "message": "Error updating profile"}), 500

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
    
    try:
        conn = mysql.connector.connect(**db_cognifit)
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
            conn.close()
            
            return jsonify({"success": True, "message": "Password changed successfully!"})
        else:
            cur.close()
            conn.close()
            return jsonify({"success": False, "message": "Current password is incorrect"}), 400
            
    except Exception as e:
        print(f"Error changing password: {e}")
        return jsonify({"success": False, "message": "Error changing password"}), 500

# -------------------------
# Calendar Route
# -------------------------
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
   
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
       
        cur.execute("""
            SELECT note_date, note_content, created_at
            FROM calendar_notes
            WHERE user_id = %s
            ORDER BY note_date DESC, created_at DESC
        """, (user_id,))
       
        notes = cur.fetchall()
        cur.close()
        conn.close()
       
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
 
@app.route('/get_upcoming_notes')
def get_upcoming_notes():
    """Get upcoming notes for notifications"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
   
    try:
        user_id = session['user_id']
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
       
        # Get today's date and next 7 days
        today = datetime.now().date()
        next_week = today + timedelta(days=7)
       
        cur.execute("""
            SELECT note_date, note_content, created_at
            FROM calendar_notes
            WHERE user_id = %s AND note_date BETWEEN %s AND %s
            ORDER BY note_date ASC, created_at ASC
        """, (user_id, today, next_week))
       
        upcoming_notes = cur.fetchall()
       
        cur.close()
        conn.close()
       
        # Format the data
        notes_by_date = {}
        for note in upcoming_notes:
            date_str = note['note_date'].strftime('%Y-%m-%d')
            if date_str not in notes_by_date:
                notes_by_date[date_str] = []
            notes_by_date[date_str].append({
                'content': note['note_content'],
                'timestamp': note['created_at'].isoformat() if note['created_at'] else None
            })
       
        return jsonify({
            "success": True,
            "upcoming_notes": notes_by_date,
            "today": today.strftime('%Y-%m-%d')
        })
       
    except Exception as e:
        print(f"Error fetching upcoming notes: {e}")
        return jsonify({"success": False, "message": "Error fetching upcoming notes"}), 500
 
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
   
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor()
       
        # Insert the note with user_id
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
        conn.close()
       
        return jsonify({
            "success": True,
            "message": "Note saved successfully!",
            "timestamp": created_at.isoformat()
        })
       
    except Exception as e:
        print(f"Error saving calendar note: {e}")
        return jsonify({"success": False, "message": "Error saving note"}), 500
# -------------------------
# 
# Blog Routes
# -------------------------
# -------------------------
# Blog Routes
# -------------------------
 
@app.route('/get_blog/<int:blog_id>')
def get_blog(blog_id):
    """Get single blog for modal"""
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
       
        cur.execute("SELECT * FROM blogs WHERE id = %s", (blog_id,))  # Remove the status filter
        blog = cur.fetchone()
       
        cur.close()
        conn.close()
       
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
                    "is_featured": blog['is_featured'],
                    "status": blog['status'],
                    "user_id": blog['user_id'],  # ADD THIS LINE
                    "created_at": blog['created_at'].strftime('%b %d, %Y') if blog['created_at'] else ''
                }
            })
        else:
            return jsonify({"success": False, "message": "Blog not found"}), 404
           
    except Exception as e:
        print(f"Error fetching blog: {e}")
        return jsonify({"success": False, "message": "Error fetching blog"}), 500
 
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
        user_id = session['user_id']  # Get the current user's ID
       
        # Validate required fields
        if not all([title, content, category, author, read_time]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400
       
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor()
       
        # If setting as featured, unfeature other blogs
        if is_featured:
            cur.execute("UPDATE blogs SET is_featured = FALSE WHERE is_featured = TRUE")
       
        # Insert new blog WITH user_id
        cur.execute("""
            INSERT INTO blogs (title, content, excerpt, category, category_color, author, read_time, image_url, is_featured, status, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (title, content, excerpt, category, category_color, author, read_time, image_url, is_featured, status, user_id))
       
        conn.commit()
        blog_id = cur.lastrowid
       
        cur.close()
        conn.close()
       
        return jsonify({
            "success": True,
            "message": "Blog created successfully!",
            "blog_id": blog_id
        })
       
    except Exception as e:
        print(f"Error creating blog: {e}")
        return jsonify({"success": False, "message": "Error creating blog"}), 500
 
@app.route('/admin/update_blog/<int:blog_id>', methods=['POST'])
def update_blog(blog_id):
    """Update a blog"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
   
    try:
        data = request.get_json()
        user_id = session['user_id']
       
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
       
        # First, check if the current user owns this blog
        cur.execute("SELECT user_id FROM blogs WHERE id = %s", (blog_id,))
        blog = cur.fetchone()
       
        if not blog:
            return jsonify({"success": False, "message": "Blog not found"}), 404
       
        # Convert both to integers for proper comparison
        blog_user_id = int(blog['user_id']) if blog['user_id'] is not None else 0
        session_user_id = int(user_id)
       
        print(f"DEBUG: Blog user_id: {blog_user_id} (type: {type(blog_user_id)})")
        print(f"DEBUG: Session user_id: {session_user_id} (type: {type(session_user_id)})")
        print(f"DEBUG: Match: {blog_user_id == session_user_id}")
       
        # Check ownership - only allow if user owns the blog
        if blog_user_id != session_user_id:
            return jsonify({"success": False, "message": "You can only edit your own blogs"}), 403
       
        # Rest of your update code...
       
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
        conn.close()
       
        return jsonify({"success": True, "message": "Blog updated successfully!"})
       
    except Exception as e:
        print(f"Error updating blog: {e}")
        return jsonify({"success": False, "message": "Error updating blog"}), 500
 
@app.route('/admin/delete_blog/<int:blog_id>', methods=['DELETE'])
def delete_blog(blog_id):
    """Delete a blog"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
   
    try:
        user_id = session['user_id']
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor()
       
        # First, check if the current user owns this blog
        cur.execute("SELECT user_id FROM blogs WHERE id = %s", (blog_id,))
        blog = cur.fetchone()
       
        if not blog:
            return jsonify({"success": False, "message": "Blog not found"}), 404
       
        # Check ownership - only allow if user owns the blog
        if blog[0] != user_id:
            return jsonify({"success": False, "message": "You can only delete your own blogs"}), 403
       
        cur.execute("DELETE FROM blogs WHERE id = %s", (blog_id,))
        conn.commit()
       
        cur.close()
        conn.close()
       
        return jsonify({"success": True, "message": "Blog deleted successfully!"})
       
    except Exception as e:
        print(f"Error deleting blog: {e}")
        return jsonify({"success": False, "message": "Error deleting blog"}), 500
@app.route('/admin/get_blogs')
def admin_get_blogs():
    """Get all blogs for admin panel"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
   
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
       
        # Make sure this query includes user_id
        cur.execute("SELECT *, COALESCE(user_id, 0) as user_id FROM blogs ORDER BY created_at DESC")
        blogs = cur.fetchall()
       
        cur.close()
        conn.close()
       
        # Convert datetime objects to strings
        for blog in blogs:
            if blog['created_at']:
                blog['created_at'] = blog['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if blog['updated_at']:
                blog['updated_at'] = blog['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
       
        return jsonify({"success": True, "blogs": blogs})
       
    except Exception as e:
        print(f"Error fetching blogs: {e}")
        return jsonify({"success": False, "message": "Error fetching blogs"}), 50
# -------------------------
# Workout Routes
# -------------------------

@app.route('/workout')
def workout():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('workout.html', 
                         user_name=session.get('username', 'User'))

@app.route('/api/workouts', methods=['GET'])
def get_workouts():
    """Get all workouts for the logged-in user"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
        
        cur.execute("""
            SELECT * FROM workouts 
            WHERE user_id = %s 
            ORDER BY workout_date DESC, created_at DESC
        """, (user_id,))
        
        workouts = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert datetime objects to strings
        for workout in workouts:
            if workout['workout_date']:
                workout['workout_date'] = workout['workout_date'].strftime('%Y-%m-%d')
            if workout['created_at']:
                workout['created_at'] = workout['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({"success": True, "workouts": workouts})
        
    except Exception as e:
        print(f"Error fetching workouts: {e}")
        return jsonify({"success": False, "message": "Error fetching workouts"}), 500

@app.route('/api/workouts', methods=['POST'])
def create_workout():
    """Create a new workout"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    data = request.get_json()
    
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400
    
    workout_type = data.get('workout_type')
    workout_date = data.get('workout_date')
    duration_minutes = data.get('duration_minutes')
    intensity_level = data.get('intensity_level')
    notes = data.get('notes', '')
    
    # Validate required fields
    if not all([workout_type, workout_date, duration_minutes, intensity_level]):
        return jsonify({"success": False, "message": "Missing required fields"}), 400
    
    # Calculate calories burned
    calories_burned = calculate_calories_burned(user_id,workout_type, duration_minutes, intensity_level)
    
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor()
        
        # Insert new workout
        cur.execute("""
            INSERT INTO workouts (user_id, workout_type, workout_date, duration_minutes, intensity_level, calories_burned, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_id, workout_type, workout_date, duration_minutes, intensity_level, calories_burned, notes))
        
        conn.commit()
        workout_id = cur.lastrowid
        
        cur.close()
        conn.close()
        
        return jsonify({
            "success": True, 
            "message": "Workout logged successfully!",
            "workout_id": workout_id,
            "calories_burned": calories_burned
        })
        
    except Exception as e:
        print(f"Error creating workout: {e}")
        return jsonify({"success": False, "message": "Error creating workout"}), 500

@app.route('/api/workouts/<int:workout_id>', methods=['DELETE'])
def delete_workout(workout_id):
    """Delete a workout"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor()
        
        # Delete workout (only if it belongs to the user)
        cur.execute("DELETE FROM workouts WHERE id = %s AND user_id = %s", (workout_id, user_id))
        conn.commit()
        
        deleted_count = cur.rowcount
        cur.close()
        conn.close()
        
        if deleted_count > 0:
            return jsonify({"success": True, "message": "Workout deleted successfully!"})
        else:
            return jsonify({"success": False, "message": "Workout not found or access denied"}), 404
        
    except Exception as e:
        print(f"Error deleting workout: {e}")
        return jsonify({"success": False, "message": "Error deleting workout"}), 500

@app.route('/api/workouts/stats', methods=['GET'])
def workout_stats_api():
    """Get workout statistics for the logged-in user"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    
    user_id = session['user_id']
    stats = get_workout_stats(user_id)  # Calls your helper function correctly
    
    return jsonify({
        "success": True,
        "stats": stats
    })


# -------------------------
# Menstrual Cycle Tracking - FIXED ROUTES
# -------------------------
@app.route('/cycle')
def cycle_form():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if user has existing cycle data - FILTERED BY USER_ID
    user_id = session['user_id']
    existing_data = None
    
    try:
        conn = mysql.connector.connect(**db_cognifit)
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
        conn.close()
    except mysql.connector.Error as err:
        print(f"DEBUG: Error fetching existing cycle data: {err}")
    
    # Pass current date to template for max date validation
    current_date = datetime.now().strftime("%Y-%m-%d")
    return render_template('cycle.html', existing_data=existing_data, current_date=current_date)


# -------------------------
# Cycle History Route
# -------------------------
@app.route('/cycle_history')
def cycle_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    cycle_history = []
    
    try:
        conn = mysql.connector.connect(**db_cognifit)
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT last_period, cycle_length, period_duration, created_at
            FROM menstrual_cycle 
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        cycle_history = cur.fetchall()
        cur.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"DEBUG: Error retrieving cycle history: {err}")
        flash("Error retrieving cycle history.", "error")
    
    return render_template('cycle_history.html', cycle_history=cycle_history, user_name=session.get('username'))


# -------------------------
# Logout
# -------------------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('user_email', None)
    return redirect(url_for('login'))

# ✅ Workout Progress Summary API
@app.route('/api/progress/summary')
def progress_summary():
    conn = None  # ✅ initialize conn so 'finally' won't crash

    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"success": False, "message": "User not logged in"}), 401

        # ✅ Handle the "all" option safely
        range_param = request.args.get('range', '7')
        if range_param == 'all':
            range_days = None  # means no filter
        else:
            range_days = int(range_param)

        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='cognifit'
        )
        cursor = conn.cursor(dictionary=True)

        if range_days:
            start_date = datetime.now() - timedelta(days=range_days)
            cursor.execute("""
                SELECT DATE(workout_date) AS workout_date,
                       SUM(duration_minutes) AS duration_minutes,
                       SUM(calories_burned) AS calories_burned
                FROM workouts
                WHERE user_id = %s AND workout_date >= %s
                GROUP BY DATE(workout_date)
                ORDER BY DATE(workout_date)
            """, (user_id, start_date))
        else:
            # "All Time" – no date filter
            cursor.execute("""
                SELECT DATE(workout_date) AS workout_date,
                       SUM(duration_minutes) AS duration_minutes,
                       SUM(calories_burned) AS calories_burned
                FROM workouts
                WHERE user_id = %s
                GROUP BY DATE(workout_date)
                ORDER BY DATE(workout_date)
            """, (user_id,))

        workout_data = cursor.fetchall()

        # Frequency data
        if range_days:
            cursor.execute("""
                SELECT DATE(workout_date) AS period, COUNT(*) AS workout_count
                FROM workouts
                WHERE user_id = %s AND workout_date >= %s
                GROUP BY DATE(workout_date)
                ORDER BY DATE(workout_date)
            """, (user_id, start_date))
        else:
            cursor.execute("""
                SELECT DATE(workout_date) AS period, COUNT(*) AS workout_count
                FROM workouts
                WHERE user_id = %s
                GROUP BY DATE(workout_date)
                ORDER BY DATE(workout_date)
            """, (user_id,))

        frequency_data = cursor.fetchall()

        return jsonify({
            "success": True,
            "chart_data": {
                "workout_data": workout_data,
                "frequency_data": frequency_data
            }
        })

    except Exception as e:
        print("Error in /api/progress/summary:", e)
        return jsonify({"success": False, "message": str(e)})

    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)