from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# --- Load the Trained Model ---
# We wrap this in a try-block to give a clear error if the model is missing
try:
    model = joblib.load('models/model_rf.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'models/model_rf.pkl' not found.")
    print("Please run 'project_analysis.ipynb' first to generate the model.")
    exit()

@app.route('/')
def home():
    """
    Renders the main page (index.html).
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission, makes a prediction, 
    and returns the result to the page.
    """
    if request.method == 'POST':
        try:
            # 1. Extract data from the form
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            height = int(request.form['height'])
            weight = float(request.form['weight'])
            ap_hi = int(request.form['ap_hi'])
            ap_lo = int(request.form['ap_lo'])
            cholesterol = int(request.form['cholesterol'])
            gluc = int(request.form['gluc'])
            smoke = int(request.form['smoke'])
            alco = int(request.form['alco'])
            active = int(request.form['active'])

            # 2. Create a DataFrame for the model
            # The column names MUST match the training data exactly
            input_data = pd.DataFrame([[
                age, gender, height, weight, ap_hi, ap_lo, 
                cholesterol, gluc, smoke, alco, active
            ]], columns=[
                'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                'cholesterol', 'gluc', 'smoke', 'alco', 'active'
            ])

            # 3. Make the Prediction
            # prediction will be 0 (Healthy) or 1 (Disease)
            prediction = model.predict(input_data)[0]
            
            # Get the probability (confidence score)
            probability = model.predict_proba(input_data)[0][1] * 100

            # 4. Determine Result Text and Color
            if prediction == 1:
                result_text = f"Warning: High Risk of Heart Disease Detected ({probability:.1f}% chance)"
                result_color = "#e74c3c"  # Red color for danger
            else:
                result_text = f"Result: Low Risk of Heart Disease ({probability:.1f}% chance)"
                result_color = "#2ecc71"  # Green color for safety

            # 5. Send result back to HTML
            return render_template('index.html', prediction_text=result_text, color=result_color)

        except Exception as e:
            # If something goes wrong (e.g., bad input), show error
            return render_template('index.html', prediction_text=f"Error: {str(e)}", color="black")

if __name__ == "__main__":
    # Run the app in debug mode
    app.run(debug=True)