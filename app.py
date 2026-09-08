from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Model file mappings
MODEL_FILES = {
    'logistic_regression_smote': 'log_reg_smote_model.pkl',
    'logistic_regression_class_weight': 'weighted_model.pkl',
    'random_forest_smote': 'rf_smote_model.pkl',
    'random_forest_class_weight': 'rf_weight.pkl',
    'xgboost_smote': 'xgb_smote_model.json',
    'xgboost_class_weight': 'xgb_weight.json',
}

models = {}

# Load pre-trained models
for key, filename in MODEL_FILES.items():
  try:
    if filename.endswith('.json'):
      clf = xgb.XGBClassifier()
      clf.load_model(filename)
      models[key] = clf
      print(f'Loaded {filename} via XGBoost successfully.')
    else:
      models[key] = joblib.load(filename)
      print(f'Loaded {filename} via Joblib successfully.')
  except Exception as e:
    print(f'Warning: Could not load {filename}: {e}')

# Column Name Mapping: Flask Form Names -> Colab Model Names
COLUMN_MAPPING = {
    'Admission Method Description': 'Admission Method Encoded',
    'Discharge Method Description': 'Discharge Method Encoded',
    'Type of Patient Description': 'Patient Type Encoded',
    'Main Specialty Description': 'Main Specialty Encoded',
    'Admitting Ward Description': 'Admitting Ward Category Encoded',
    'Length of Stay (Days)': 'Length_of_Stay_Encoded',
    'Primary Diagnosis Code': 'Main_Diagnosis_Group_Encoded',
}

# Exact 17 features order expected by Colab trained models
EXPECTED_FEATURE_ORDER = [
    'Gender',
    'Ethnic Group Description',
    'Age',
    'Discharge Alert Status',
    'Medications',
    'Lives alon',
    'SCC',
    'Follow-up',
    'Comorbidities',
    'Discharge Destination Address',
    'Admission Method Encoded',
    'Discharge Method Encoded',
    'Patient Type Encoded',
    'Main Specialty Encoded',
    'Admitting Ward Category Encoded',
    'Length_of_Stay_Encoded',
    'Main_Diagnosis_Group_Encoded',
]

# Standard decision threshold (50%) for binary classification
HIGH_RISK_THRESHOLD = 0.50
MODERATE_RISK_THRESHOLD = 0.30


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/about')
def about():
  return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.method == 'POST':
    # 1. Model & Methodology Selection
    model_choice = request.form.get('model_choice', 'random_forest')
    resampling_method = request.form.get('resampling_method', 'smote')
    model_key = f'{model_choice}_{resampling_method}'
    selected_model = models.get(model_key)

    # 2. Extract Encoded Feature Inputs from Form
    feature_data = {
        'Gender': int(request.form['gender']),
        'Ethnic Group Description': int(request.form['ethnic_group']),
        'Age': float(request.form['age']),
        'Discharge Alert Status': int(request.form['discharge_alert']),
        'Medications': float(request.form['medications']),
        'Lives alon': int(request.form['lives_alone']),
        'SCC': int(request.form['scc']),
        'Follow-up': int(request.form['follow_up']),
        'Comorbidities': float(request.form['comorbidities']),
        'Discharge Destination Address': int(
            request.form['discharge_destination_address']
        ),
        'Admission Method Description': int(request.form['admission_method']),
        'Discharge Method Description': int(request.form['discharge_method']),
        'Type of Patient Description': int(request.form['patient_type']),
        'Main Specialty Description': int(request.form['main_specialty']),
        'Admitting Ward Description': int(request.form['admitting_ward']),
        'Length of Stay (Days)': float(request.form['length_of_stay']),
        'Primary Diagnosis Code': int(request.form['primary_diagnosis_code']),
    }

    # Convert to DataFrame & rename columns to match Colab model training
    input_df = pd.DataFrame([feature_data])
    predict_df = input_df.rename(columns=COLUMN_MAPPING)

    # Reorder columns explicitly to match exact training structure
    predict_df = predict_df[EXPECTED_FEATURE_ORDER]

    # Convert to NumPy array to prevent feature name warnings/mismatches
    input_matrix = predict_df.values

    # 3. Predict Readmission Probability & Apply Risk Logic
    if selected_model is not None:
      if hasattr(selected_model, 'predict_proba'):
        proba_array = selected_model.predict_proba(input_matrix)
        proba_class_1 = float(proba_array[0][1])
        proba_class_0 = float(proba_array[0][0])
      else:
        proba_class_1 = float(selected_model.predict(input_matrix)[0])
        proba_class_0 = 1.0 - proba_class_1

      probability = round(proba_class_1 * 100, 2)
      no_readmission_probability = round(proba_class_0 * 100, 2)

      # Risk Classification Logic
      if proba_class_1 >= HIGH_RISK_THRESHOLD:
        prediction = 1
        risk_status = 'High Risk – Readmission Likely'
        badge_class = 'bg-danger'
      elif proba_class_1 >= MODERATE_RISK_THRESHOLD:
        prediction = 0
        risk_status = 'Moderate Risk – Monitor Patient'
        badge_class = 'bg-warning text-dark'
      else:
        prediction = 0
        risk_status = 'Low Risk – Not Readmitted'
        badge_class = 'bg-success'
    else:
      prediction = 0
      probability = 0.0
      no_readmission_probability = 100.0
      risk_status = 'Model Unavailable'
      badge_class = 'bg-secondary'

    result = {
        'prediction': prediction,
        'readmission_probability': probability,
        'no_readmission_probability': no_readmission_probability,
        'risk_status': risk_status,
        'badge_class': badge_class,
        'model_used': model_choice.replace('_', ' ').title(),
        'methodology_used': resampling_method.replace('_', ' ').upper(),
    }

    return render_template('result.html', result=result)

  return render_template('predict.html')

@app.route('/data-stats')
def data_stats():
  return render_template('data_stats.html')

if __name__ == '__main__':
  app.run(debug=True)