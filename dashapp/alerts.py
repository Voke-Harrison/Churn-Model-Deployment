import pandas as pd
import os
from datetime import datetime


def get_churn_alert_message(file_path='uploads/predicted_churn_results.xlsx'):
    try:
        if not os.path.exists(file_path):
            return "No prediction file found."

        df = pd.read_excel(file_path)

        if "Predicted Churn" not in df.columns:
            return "Missing 'Predicted Churn' column."

        churned = df[df["Predicted Churn"] == "Churned"]
        if not churned.empty:
            return f"🚨 {len(churned)} customers are at high risk of churn."
        else:
            return "✅ No high-risk churn customers currently."

    except Exception as e:
        return f"Error reading alerts: {str(e)}"


def export_churned_customers(file_path='uploads/predicted_churn_results.xlsx'):
    try:
        if not os.path.exists(file_path):
            return None

        df = pd.read_excel(file_path)
        if "Predicted Churn" not in df.columns:
            return None

        churned = df[df["Predicted Churn"] == "Churned"]
        if churned.empty:
            return None

        output_file = "uploads/at_risk_customers_only.csv"
        churned.to_csv(output_file, index=False)
        return output_file

    except Exception:
        return None


