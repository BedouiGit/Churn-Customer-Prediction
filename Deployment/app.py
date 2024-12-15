import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, scaler, and encoded columns
model = pickle.load(open('Churn_Prediction_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoded_columns = pickle.load(open('encoded_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gathering inputs from the form
        account_length = int(request.form.get('account_length'))
        international_plan = int(request.form.get('international_plan'))  # Binary: 0 or 1
        voice_mail_plan = int(request.form.get('voice_mail_plan'))  # Binary: 0 or 1
        number_vmail_messages = int(request.form.get('number_vmail_messages'))
        total_day_minutes = float(request.form.get('total_day_minutes'))
        total_day_charge = float(request.form.get('total_day_charge'))
        total_eve_minutes = float(request.form.get('total_eve_minutes'))
        total_eve_charge = float(request.form.get('total_eve_charge'))
        total_night_minutes = float(request.form.get('total_night_minutes'))
        total_intl_minutes = float(request.form.get('total_intl_minutes'))
        total_intl_calls = int(request.form.get('total_intl_calls'))
        total_intl_charge = float(request.form.get('total_intl_charge'))
        customer_service_calls = int(request.form.get('customer_service_calls'))

        # Create DataFrame for inputs
        inputs = pd.DataFrame(
            [[
                account_length, international_plan, voice_mail_plan,
                number_vmail_messages, total_day_minutes, total_day_charge,
                total_eve_minutes, total_eve_charge, total_night_minutes,
                total_intl_minutes, total_intl_calls, total_intl_charge,
                customer_service_calls
            ]],
            columns=[
                'Account length', 'International plan', 'Voice mail plan',
                'Number vmail messages', 'Total day minutes', 'Total day charge',
                'Total eve minutes', 'Total eve charge', 'Total night minutes',
                'Total intl minutes', 'Total intl calls', 'Total intl charge',
                'Customer service calls'
            ]
        )

        # Align the input DataFrame with encoded columns
        inputs = inputs.reindex(columns=encoded_columns, fill_value=0)

        # Scale the numeric inputs
        input_scaled = scaler.transform(inputs)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Generate churn risk scores
        churn_risk_scores = np.round(model.predict_proba(input_scaled)[:, 1] * 100, 2)

        # Churn flag
        prediction_text = 'YES' if prediction[0] == 1 else 'NO'

        # Return the result to the template
        return render_template(
            'predict.html',
            prediction=prediction_text,
            churn_risk_scores=churn_risk_scores[0],
            inputs=request.form
        )

    except Exception as e:
        return render_template(
            'predict.html',
            prediction='Error',
            churn_risk_scores='N/A',
            error=str(e)
        )

if __name__ == '__main__':
    app.run(debug=True)

    