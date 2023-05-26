import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # Splitting
from sklearn.model_selection import cross_val_score # Cross-Validation
from sklearn.ensemble import RandomForestClassifier # Model
from sklearn import metrics # Metrics
from flask import Flask, jsonify, request
import json
import joblib
import warnings
warnings.filterwarnings("ignore")


response = ''
app = Flask(__name__)
@app.route('/name', methods = ["GET","POST"])

def nameRoute():

    global response

    if (request.method == "POST"):
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        name = request_data['name']
        sym = np.array(name)
        sym = np.array([sym])        

        # Model Accuracy, how often is the classifier correct?
        #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        #print(cross_val_score(rnd_forest,X_train_data,y_train_data,cv=5).mean())
        with open('Model.joblib', 'rb') as f:
            predictor = joblib.load(f)
        value = predictor.predict(sym)
        prob = predictor.predict_proba(sym)
        class_labels = predictor.classes_
        disease = value[0]
        class_probabilities = dict(zip(class_labels, prob[0]))
        response = class_probabilities
        
        # with open('model.joblib', 'rb') as f:
        #     predictor = joblib.load(f)
        # model_json = rnd_forest.to_json()
        # with open("model.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # rnd_forest.save_weights("model.h5")
        # print("Saved model to disk")
        return ""
    else:
        return jsonify({'name' : response})

if __name__ == "__main__" :
    app.run(debug=False)

