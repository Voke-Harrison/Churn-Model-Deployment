# 🧠 Customer Churn Prediction Web App

This project is a full-stack web application built with **Flask** and **Dash**, allowing users to perform both **single** and **bulk customer churn predictions** using a trained machine learning model. It also provides **downloadable reports** and an interactive **dashboard** for insights.

---

## 🚀 Features

- 🎯 **Churn Prediction**: Predict individual or bulk customer churn using a trained GBDT model.
- 📈 **Interactive Dashboards**: Visualize churn risk, revenue impact, tenure insights, city-level churn, and customer segmentation.
- 🛎️ **Real-time Alerts**: View alerts when high churn risk thresholds are exceeded.
- 📥 **Downloadable Reports**: Export churn predictions and filtered datasets.
- 📡 **Live Dashboard Updates**: Automatically refresh dashboards using new data (optional).
- 📦 **Modular Architecture**: Separates model, dashboard, alerts, and main app logic for easier maintenance.

---

## 🧰 Tech Stack

- Python, Flask
- Scikit-learn, Pandas, Numpy
- Dash + Plotly
- HTML/CSS (Bootstrap)
- Excel file handling with `openpyxl`

---

## 📁 Project Structure


├── app.py # Main Flask application
├── gbdt_model.pkl
├── gbdt_features.pkl
├── dashapp/
│   ├── dyna_dashboard.py # Dash app layout and callbacks
│   ├── alerts.py # Churn alert logic
├── templates/
│   └── index.html # Frontend HTML for the prediction app
│   └── charts.html # Frontend HTML charts
├── logs/
│   └── predictions_log.csv # keeps logs of the app
├── uploads/ # Uploads directory (Excel and CSV files)
│   └── predicted_churn_results.xlsx # Output of Bulk predictions
│   └── sample_bulk_upload.xlsx # the uploaded file
│   └── at_risk_customers_only.csv # At risk customers only
├── static/
├   └── sample_bulk_upload.xlsx # Sample file for  upload
│   └── css/
│       └── style.css
├── All Project Documents
│   └── Taco-Tel Churn Prediction Data Exploration.ipynb
│   └── Taco-Tel Churn Prediction Model.ipynb
│   └── Taco-Tel Churn Prediction Reports and Dashboard.ipynb
│   └── BAN6800 Module 1 Assignment - Voke H Edafejimue - 1443304
│   └── BAN6800 Module 2 Assignment - Voke H Edafejimue - 1443304
│   └── BAN6800 - Milestone 1 - Voke H Edafejimue - 1443304
│   └── BAN6800 - Module 4 - Voke H Edafejimue - 1443304
│   └── BAN6800 - Milestone 2 - Voke H Edafejimue - 1443304
├── requirements.txt 
├── Procfile
├── .gitignore
└── README.md

---

## 🛠️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/churn-predictor-app.git
cd churn-predictor-app

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. **Install dependencies**

```bash
pip install -r requirements.txt

4. **Add your model and feature files**

Place the following in the project folder:

Your trained model: gbdt_model.pkl
Feature list used during training: gbdt_features.pkl

---
## ▶️ Run Locally

```bash
python app.py

- Visit http://localhost:5000 in your browser.

## 🚀 Deployment on Heroku
- Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
- Login to Heroku

```bash
heroku login

- Create Heroku App

```bash
heroku create churn-predictor-app

Add files for deployment
Make sure you have these:

app.py
dash_dashboard.py
Procfile
requirements.txt
runtime.txt (optional)

- Initialize Git and push

```bash
git init
git add .
git commit -m "Initial commit"
git push heroku master

- Open your deployed app

```bash
heroku open

## 📌 Notes
- Ensure your input features match the model’s training features.
- Use the sample Excel template (sample_template.xlsx) for bulk uploads.
- This app uses a simple Gradient Boosting model. Swap out with your preferred model if needed.
