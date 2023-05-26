import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, jsonify, request
import json
import joblib
import warnings
import secrets

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'your_secret_key'
user_responses = {}

@app.route('/name', methods=["GET", "POST"])
def nameRoute():
    if request.method == "POST":
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        name = request_data['name']
        sym = np.array(name)
        sym = np.array([sym])

        with open('Model.joblib', 'rb') as f:
            predictor = joblib.load(f)
        
        value = predictor.predict(sym)
        prob = predictor.predict_proba(sym)
        class_labels = predictor.classes_
        disease = value[0]
        class_probabilities = dict(zip(class_labels, prob[0]))
        
        # Generate a unique ID for the user
        user_id = secrets.token_hex(16)
        user_responses[user_id] = class_probabilities
        
        return jsonify({'user_id': user_id})
    
    else:
        user_id = request.args.get('user_id')
        
        if user_id in user_responses:
            response = user_responses[user_id]
            return jsonify({'name': response})
        else:
            return jsonify({'error': 'Invalid user_id'})

if __name__ == "__main__":
    app.run(debug=False)
