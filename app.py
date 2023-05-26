import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, jsonify, request, session
import json
import joblib
import warnings
import secrets

warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/name', methods=["GET", "POST"])
def nameRoute():
    if request.method == "POST":
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        name = request_data['name']
        sym = np.array(name)
        sym = np.array([sym])

        # Model Accuracy, how often is the classifier correct?
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        # print(cross_val_score(rnd_forest,X_train_data,y_train_data,cv=5).mean())
        with open('Model.joblib', 'rb') as f:
            predictor = joblib.load(f)
        value = predictor.predict(sym)
        prob = predictor.predict_proba(sym)
        class_labels = predictor.classes_
        disease = value[0]
        class_probabilities = dict(zip(class_labels, prob[0]))
        response = class_probabilities
        return jsonify({'name': response})

if __name__ == "__main__":
    app.run(debug=False)
