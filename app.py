import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
from sklearn.impute import SimpleImputer
import dash
import dash.html as html
import dash.dcc as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import logging
from datetime import datetime
from flask_socketio import SocketIO, emit
import dash
from dash import Dash, dcc, html
from dashapp.dyna_dashboard import init_dashboard
from dashapp.alerts import get_churn_alert_message, export_churned_customers


# Configure logging
logging.basicConfig(
    filename='logs/predictions_log.csv',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

socketio = SocketIO(app)

#dash_app = create_dash_app(app)
dash_app = init_dashboard(app)

# Load the saved model and features file
model_file = './gbdt_model.pkl'
features_file = './gbdt_features.pkl'

with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Load the features (list of feature names)
with open(features_file, 'rb') as f:
    features = pickle.load(f)


# Define allowed file extensions for bulk upload
ALLOWED_EXTENSIONS = {'xlsx'}

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to handle missing values
def handle_missing_values(df):
    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Impute numerical features with the mean
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    # Impute categorical features with the mode
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])

    return df

# Define the categorical mapping function
def map_categorical_features(df):
    categorical_features = ['Gender', 'Partner', 'Online Security', 'Paperless Billing', 'Dependents']
    categorical_mapping = {
        'Gender': {'Male': 0, 'Female': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Online Security': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'Paperless Billing': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1}
    }
    
    # Map categorical variables to numeric values
    for col in categorical_features:
        if col in df.columns:
            if col in categorical_mapping:
                # Ensure that we map all possible values, and fill any NaN with 0
                df[col] = df[col].map(categorical_mapping[col]).fillna(0)
            else:
                # If mapping is not defined, we need to check the unique values to apply manual mapping
                print(f"Warning: No mapping for column {col}. Values: {df[col].unique()}")
                df[col] = df[col].fillna(0)    
    return df

# Home Route
@app.route('/')
def index():
    churn_counts = {'Churned': 1, 'Not Churned': 0}
    try:
        df = pd.read_csv('logs/predictions_log.csv')
        churn_counts = df['Prediction'].value_counts().to_dict()
    except Exception as e:
        print("Error loading prediction log for charts:", e)

    alert_message = "You have no active alerts. Click the Yellow Button."
    return render_template('index.html', churn_counts=churn_counts, alert_message=alert_message)
    
# Single Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for feature in features:
            input_data[feature] = request.form.get(feature, 0)  # Default to 0 if not provided

        input_df = pd.DataFrame([input_data])
        input_df = handle_missing_values(input_df)
        input_df = input_df[features]
        
        # Apply the mapping to categorical columns
        input_df = map_categorical_features(input_df)
        
        prediction = model.predict(input_df)
        result = 'Churned' if prediction[0] == 1 else 'Not Churned'

        # Write to prediction log CSV
        log_entry = pd.DataFrame([{
            'timestamp': datetime.now(),
            'input_data': json.dumps(input_data),
            'prediction': result
        }])
        os.makedirs('logs', exist_ok=True)
        log_path = 'logs/predictions_log.csv'
        log_entry.to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))

        # Log the prediction input and result
        logging.info(f"SINGLE Prediction | Input: {input_data} | Result: {result}")
        socketio.emit('new_prediction', {'prediction': result})
        return jsonify({'prediction': result})

    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)})


# Bulk Prediction Route
@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            df = pd.read_excel(filepath)

            missing_features = set(features) - set(df.columns)
            if missing_features:
                return jsonify({'error': f'Missing features in input data: {", ".join(missing_features)}'})

            df = df[features]
            df = handle_missing_values(df)

            # Apply the mapping to categorical columns
            df = map_categorical_features(df)

            predictions = model.predict(df)
            df['Predicted Churn'] = ['Churned' if pred == 1 else 'Not Churned' for pred in predictions]

            # Write to prediction log CSV
            os.makedirs('logs', exist_ok=True)
            log_path = 'logs/predictions_log.csv'

            bulk_logs = pd.DataFrame([{
                'timestamp': datetime.now(),
                'input_data': row.drop('Predicted Churn').to_json(),
                'prediction': row['Predicted Churn']
            } for _, row in df.iterrows()])

            bulk_logs.to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))

            # Log each row prediction
            for i, row in df.iterrows():
                row_data = row.to_dict()
                logging.info(f"BULK Prediction | Input: {row_data}")

            output_file = os.path.join('uploads', 'predicted_churn_results.xlsx')
            df.to_excel(output_file, index=False)

            return jsonify({'message': 'Bulk prediction completed', 'download_url': '/uploads'})

        return jsonify({'error': 'Invalid file format or no file uploaded'})

    except Exception as e:
        logging.error(f"Error in /bulk_predict: {str(e)}")
        return jsonify({'error': str(e)})

# Download Predictions Route
@app.route('/download_predictions')
def download_predictions():
    # Path to the generated file
    file_path = os.path.join('uploads', 'predicted_churn_results.xlsx')
    
    # Send the file to the user
    return send_file(file_path, as_attachment=True)

@app.route('/logs')
def view_logs():
    try:
        with open('logs/predictions_log.csv', 'r') as f:
            content = f.read().replace('\n', '<br>')  # Replace newlines with HTML line breaks
        return f"<h2>Prediction Logs</h2><div style='font-family: monospace;'>{content}</div>"
    except Exception as e:
        return f"<h3>Error reading log file: {str(e)}</h3>"
    
@app.route('/download_logs')
def download_logs():
    try:
        return send_file('logs/predictions_log.csv', as_attachment=True)
    except Exception as e:
        return f"<h3>Error downloading log file: {str(e)}</h3>"    
    
@app.route('/activate_alerts', methods=['POST'])
def activate_alerts():
    alert_message = check_alerts()  # Get the alert message
    print(f"Alert message: {alert_message}")  # Log to see what is being passed
    return render_template('index.html', alert_message=alert_message)


@app.route('/alerts_data')
def alerts_data():
    message = get_churn_alert_message()
    return jsonify({"message": str(message)})

@app.route('/download_churn_alerts')
def download_churn_alerts():
    file_path = export_churned_customers()
    if file_path and os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "No at-risk customers to download.", 204

@app.route('/charts')
def charts():
    try:
        # Read the CSV while skipping problematic lines using the Python engine
        df = pd.read_csv('logs/predictions_log.csv', quotechar='"', engine='python', on_bad_lines='skip')

        # Ensure the 'timestamp' column is parsed as datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Drop rows where 'timestamp' is invalid
        df = df.dropna(subset=['timestamp'])

        # Extract date from timestamp for grouping
        df['date'] = df['timestamp'].dt.date

        # Group by date and prediction for chart data
        chart_data = df.groupby(['date', 'prediction']).size().reset_index(name='count')

        # Create the bar chart using Plotly
        fig = px.bar(chart_data, x='date', y='count', color='prediction', barmode='group')

        # Generate HTML representation of the chart
        chart_html = fig.to_html(full_html=False)

        # Render the chart in HTML template
        return render_template_string('''
            <html>
                <head><title>Prediction Charts</title></head>
                <body>
                    <h2>Churn Prediction Over Time</h2>
                    {{ chart | safe }}
                </body>
            </html>
        ''', chart=chart_html)
    
    except Exception as e:
        return f"Error loading chart: {e}"
    

  


# Run the app
if __name__ == '__main__':
    socketio.run(app, debug=True)