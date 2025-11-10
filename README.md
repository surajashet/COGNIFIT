🧠 Cognifit – Personalized Fitness Monitoring and Nutritional Assistance Using AI

A smart health and fitness web platform that provides personalized workout tracking, nutritional analysis, and overall health insights using AI and image recognition.

🚀 How We Built This

Cognifit was developed as a full-stack AI-powered fitness and nutrition assistant designed to help users monitor, analyze, and optimize their health progress.

The project was built through an iterative approach — combining Flask for the backend, MySQL for data management, and Machine Learning models for personalized prediction and insights.

Our team focused on building a clean, data-driven system that allows users to:

Track daily nutrition and workouts

Monitor weekly and monthly progress

View AI-generated predictions (calories, nutrients, BMI, etc.)

Access dynamic graphs and reports

Engage with health blogs and resources

⚙️ Tech Stack Overview
Frontend

HTML5, CSS3, JavaScript – Core UI and interactivity

Bootstrap – Responsive design

Font Awesome & Google Fonts – Iconography and typography

AJAX / Fetch API – Dynamic data updates without reloading

Chart.js / Plotly – For progress graphs and analytics

Backend

Flask (Python) – Main web framework handling routing, logic, and APIs

MySQL – Relational database for user, food, and activity data

Werkzeug Security – Password hashing and authentication

Pickle + Scikit-Learn – For AI model loading and predictions

Numpy & Pandas – Data preprocessing and analysis

AI & Machine Learning

Random Forest Regressor – For calorie and nutrient prediction models

Image Recognition Module – Identifies food items via AI (using trained dataset with 20+ classes)

Custom Preprocessing Pipelines – For feature scaling and real-time predictions

Additional Tools

Supabase (planned for Next.js integration) – For authentication in future versions

Git + GitHub – Version control and deployment

VS Code / Jupyter Notebook – Development and model training environment

🧩 Core Features

✅ User Authentication & Onboarding

Secure sign-up/login with hashed passwords

Personalized onboarding flow with user health inputs

✅ AI-Based Food Recognition

Upload food images and get automatic calorie/nutrient estimation

✅ Workout & Nutrition Tracking

Log daily workouts and food intake

View graphs showing calories burned vs. consumed

✅ Progress Dashboard

Dynamic charts for weekly/monthly health trends

Goal tracking and personalized insights

✅ Overall Report

Summarized report showing activity frequency, food analysis, and BMI

✅ Health Blog Section

Curated wellness blogs integrated into the UI for educational engagement

✅ Admin Functionality (optional)

Ability to manage blogs, users, and AI models through backend interface

🗂️ Folder Structure
Cognifit/
│
├── app.py                  # Flask main application
├── static/                 # CSS, JS, and image files
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/              # HTML templates (Jinja2)
│   ├── dashboard.html
│   ├── login.html
│   ├── progress.html
│   ├── blogs.html
│   └── overall_report.html
├── models/                 # Pickled ML models
│   ├── food_model.pkl
│   ├── scaler.pkl
├── database/               # SQL scripts and DB backups
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignored files for version control

⚡ How It Works (Flow)

User Registration/Login
→ Enters details → Stored in MySQL

Personalized Setup
→ Inputs height, weight, activity → Saved to DB

Tracking & Logging
→ Food image upload or manual entry
→ Flask processes image → ML model predicts nutrients

Progress Dashboard
→ Flask fetches weekly data → Chart.js renders progress

Overall Report
→ Aggregates calories, workouts, BMI trends, and frequency table

Blog & Tips Section
→ Pulled from DB via Flask route /get_blog/<id>

🧠 AI Model Details

Model Used: RandomForestRegressor

Dataset: Curated dataset of ~20 food classes

Input Features: Image-based + Nutritional data (via image recognition)

Output: Predicted calories and macronutrients per serving

Performance: ~90% accuracy on test data

🧰 Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/cognifit.git
cd cognifit

2️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Database

Create a MySQL database named cognifit_db

Import database/cognifit.sql

Update credentials in app.py:

db_cognifit = {
    'host': 'localhost',
    'user': 'root',
    'password': 'yourpassword',
    'database': 'cognifit_db'
}

5️⃣ Run the Application
python app.py


Visit 👉 http://127.0.0.1:5000/

🌱 Future Enhancements

Integration with Next.js + Supabase Auth

AI chatbot for personalized health advice

Wearable data integration (Fitbit, Apple Health)

Real-time notifications and progress reminders

Community challenges and leaderboard
