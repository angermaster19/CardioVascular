from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd  # Import pandas

# Load the pre-trained models
stack_model = joblib.load('Model/stack_model.pkl')
scaler = joblib.load('Model/scaler.pkl')
poly = joblib.load('Model/poly_features.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(int(request.form['age'])*365)
        gender = int(request.form['gender'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        # Create a DataFrame with the same feature names as the model expects
        user_input = pd.DataFrame([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]],
                                  columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])

        # Apply scaling and polynomial features transformation
        scaled_data = scaler.transform(user_input)  # Ensure it's a DataFrame with the correct column names
        poly_features = poly.transform(scaled_data)

        # Get prediction from the model
        prediction = stack_model.predict(poly_features)

        # Return the prediction result
        return render_template('index.html', prediction=f'Predicted Class: {prediction[0]}', show_meter=True, prediction_value=prediction[0])

    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
