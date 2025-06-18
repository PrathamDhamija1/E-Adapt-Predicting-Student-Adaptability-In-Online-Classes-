from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the saved model and scaler with error handling
try:
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    model = None
    scaler = None

# Define the label encoder mappings
label_encoders = {}
categorical_columns = ['Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student', 'Location', 'Load-shedding', 'Financial Condition', 'Internet Type', 'Network Type', 'Class Duration', 'Self Lms', 'Device']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    if col == 'Gender':
        label_encoders[col].fit(['boy', 'girl'])
    elif col == 'Age':
        label_encoders[col].fit(['1-5', '6-10', '11-15', '16-20', '21-25', '26-30'])
    elif col == 'Education Level':
        label_encoders[col].fit(['school', 'college', 'university'])
    elif col == 'Institution Type':
        label_encoders[col].fit(['government', 'non government'])
    elif col == 'IT Student':
        label_encoders[col].fit(['yes', 'no'])
    elif col == 'Location':
        label_encoders[col].fit(['yes', 'no'])
    elif col == 'Load-shedding':
        label_encoders[col].fit(['low', 'high'])
    elif col == 'Financial Condition':
        label_encoders[col].fit(['poor', 'mid', 'rich'])
    elif col == 'Internet Type':
        label_encoders[col].fit(['wifi', 'mobile data'])
    elif col == 'Network Type':
        label_encoders[col].fit(['2g', '3g', '4g'])
    elif col == 'Class Duration':
        label_encoders[col].fit(['0-1', '1-3', '3-6', '6-10'])
    elif col == 'Self Lms':
        label_encoders[col].fit(['yes', 'no'])
    elif col == 'Device':
        label_encoders[col].fit(['tab', 'mobile', 'computer'])

# Label encoder for the target variable
adaptivity_encoder = LabelEncoder()
adaptivity_encoder.fit(['Low', 'Moderate', 'High'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/adapt')
def adapt():
    return render_template('adaptivity.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return render_template('result.html', adaptivity_level="Error: Model or scaler not loaded. Please check server logs.")

        # Get form data
        form_data = {
            'Gender': request.form['gender'],
            'Age': request.form['age'],
            'Education Level': request.form['education_level'],
            'Institution Type': request.form['institute_type'],
            'IT Student': request.form['it_student'],
            'Location': request.form['location'],
            'Load-shedding': request.form['load_shedding'],
            'Financial Condition': request.form['financial_condition'],
            'Internet Type': request.form['internet_type'],
            'Network Type': request.form['network_type'],
            'Class Duration': request.form['class_duration'],
            'Self Lms': request.form['self_lms'],
            'Device': request.form['device']
        }

        # Validate Class Duration format (e.g., "0-1", "1-3")
        class_duration = form_data['Class Duration']
        if '-' not in class_duration or len(class_duration.split('-')) != 2:
            return render_template('result.html', adaptivity_level=f"Error: Invalid Class Duration format: {class_duration}. Expected format: '0-1', '1-3', etc.")

        # Encode the categorical inputs
        encoded_data = []
        for col in categorical_columns:
            try:
                encoded_value = label_encoders[col].transform([form_data[col]])[0]
                encoded_data.append(encoded_value)
            except Exception as e:
                return render_template('result.html', adaptivity_level=f"Error encoding {col}: {str(e)}")

        # Convert to a DataFrame with column names to avoid UserWarning
        input_data = pd.DataFrame([encoded_data], columns=categorical_columns)

        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)

        # Decode the prediction
        adaptivity_level = adaptivity_encoder.inverse_transform(prediction)[0]

        return render_template('result.html', adaptivity_level=adaptivity_level)

    except Exception as e:
        return render_template('result.html', adaptivity_level=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)