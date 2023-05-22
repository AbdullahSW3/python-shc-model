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
        
        data=pd.read_csv("dataset.csv")
        #Getting the inforamtion about this data set and filling the Nan values with "Unknow"
        #print(data.head())
        #print(data.shape)
        data["list_of_Symps"] = 0
        for i in range(data.shape[0]):
            values = data.iloc[i].values
            values = values.tolist()
            if 0 in values:
                data["list_of_Symps"][i] = values[1:values.index(0)]
            else:
                data["list_of_Symps"][i] = values[1:]


        #print(data.tail)
        #print(data.head())


        objects = (data.dtypes == 'object')
        object_cols = list(objects[objects].index)
        #print(object_cols)
        column_values = data[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
            'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9',
            'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14',
            'Symptom_15', 'Symptom_16', 'Symptom_17']].values.ravel()
        #print(column_values)
        symptoms=pd.unique(column_values.tolist())
        #we have nan values
        symptoms = [i for i in symptoms if str(i) != "nan"]
        #print(len(symptoms))

        new_data = pd.DataFrame(columns = symptoms,index = data.index)
        #adding the disease
        new_data['list_of_Symps']=data['list_of_Symps']
        

        #Filling our data frame with 
        for i in new_data:
            new_data[i] = data.apply(lambda x:1 if i in x.list_of_Symps else 0, axis=1)


        new_data['Disease']= data['Disease']
        new_data=new_data.drop('list_of_Symps',axis=1)

        #print(new_data.head())

        #Spliting the data vset

        training_data, testing_data = train_test_split(new_data, test_size=0.2, random_state=25)
        X_train_data=training_data.drop("Disease",axis=1)
        y_train_data = training_data["Disease"].copy()
        X_test = testing_data.drop("Disease",axis=1)
        y_test = testing_data["Disease"].copy()


        #RandomForestClassifier

        rnd_forest = RandomForestClassifier()
        rnd_forest.fit(X_train_data,y_train_data)
        print(X_train_data.shape)
        print(y_train_data.shape)
        y_pred=rnd_forest.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        #print(cross_val_score(rnd_forest,X_train_data,y_train_data,cv=5).mean())

        value = rnd_forest.predict(sym)
        prob = rnd_forest.predict_proba(sym)
        class_labels = rnd_forest.classes_
        disease = value[0]
        class_probabilities = dict(zip(class_labels, prob[0]))
        response = class_probabilities
        
        joblib.dump(rnd_forest, "Model.joblib")
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

