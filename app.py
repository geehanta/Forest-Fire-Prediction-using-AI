
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('rfmodel2.pkl', 'rb'))
selected_symptoms = []  # Initialize `selected_symptoms`
prediction_made = False  # Initialize `prediction_made` to False
model_urti = pickle.load(open('rfmodelurti.pkl', 'rb'))
selected_symptoms_urti = []  # Initialize `selected_symptoms`
prediction_made_urti = False  # Initialize `prediction_made` to False

@app.route('/')
def hello_world():
    return render_template("pathogen_predict.html")

@app.route('/urti_predict')
def urti_predict():
    return render_template('urti_pathogen_predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    global selected_symptoms, prediction_made
    selected_symptoms = request.form.getlist('symptoms')

    if not selected_symptoms:
        out = "No prediction (no symptoms selected)"
    else:
        all_symptoms = ["Chills", "Cough", "Difficulty_breathing", "Sputum_production", "Sore_throat", "Headache", "Runny_nose", "Eye_pain", "Seizures", "Tick_bites", 
                        "Abdominal_pain", "Vomiting", "Diarrhoea", "Blood_in_stool", "Bleeding", "Bruising", "Rash", "Joint_aches", "Muscle_aches", "Dark_urine", "Jaundice"]
        input_features = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
        final = np.array(input_features, dtype=float)
        out = model.predict([final])[0]
        prediction_made = True  # Set the prediction flag

    return render_template('pathogen_predict.html', pred=' Answer:  {}'.format(out), prediction_made=prediction_made)

@app.route('/urti-predict', methods=['POST'])
def urtipredict():
    global selected_symptoms_urti, prediction_made_urti
    selected_symptoms_urti = request.form.getlist('symptoms_urti')

    if not selected_symptoms_urti:
        out = "No prediction (no symptoms selected)"
    else:
        all_symptoms_urti = ["Chills", "Cough", "Difficulty_breathing", "Sputum_production", "Sore_throat", "Headache", "Runny_nose", "Eye_pain", "Seizures", "Tick_bites", 
                        "Abdominal_pain", "Vomiting", "Diarrhoea", "Rash", "Joint_aches", "Muscle_aches", "Dark_urine"]
        input_features = [1 if symptom in selected_symptoms_urti else 0 for symptom in all_symptoms_urti]
        final = np.array(input_features, dtype=float)
        out = model_urti.predict([final])[0]
        prediction_made_urti = True  # Set the prediction flag

    return render_template('urti_pathogen_predict.html', urtipred=' Answer:  {}'.format(out), prediction_made_urti=prediction_made_urti)

if __name__ == '__main__':
    app.run(debug=True)

