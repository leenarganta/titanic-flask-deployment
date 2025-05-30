from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('titanic_model.pkl')

@app.route('/')
def home():
    return "The Titanic Survival Prediction API is working successfully"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    age = data['age']
    fare = data['fare']
    sex = data['sex']
    embarked = data['embarked']

    # Prediction
    features = np.array([[age, fare, sex, embarked]])
    prediction = model.predict(features)

    return jsonify({'survived': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
