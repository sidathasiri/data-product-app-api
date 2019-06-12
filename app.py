from flask import Flask, abort, request 
from flask_cors import CORS
import pandas as pd  
import numpy as np
from pandas import read_csv
import seaborn as sns  
import matplotlib.pyplot as plt  
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
import json
sns.set(style="ticks",color_codes=True)

# app = Flask(__name__)
app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')
CORS(app)
data=pd.read_csv('online_shoppers_intention.csv')

@app.route("/")
def root():
    return "works" 

@app.route("/revenue")
def revenue():
    # fig = plt.figure(figsize=(8,8))
    data['Revenue'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
    plt.title('Revenue')
    plt.legend(labels=['True','False'] )
    plt.tight_layout() 
    filePath = "./web/static/revenue.png"
    print("running")
    if os.path.exists(filePath):
        print("deleting")
        os.remove(filePath)
    plt.savefig(filePath)
    plt.clf()
    return json.dumps({"url": "http://localhost:5000/revenue.png"})

@app.route("/operating_systems")
def operatingSystems():
    plt.hist(data.OperatingSystems, bins=20)
    plt.ylabel('Operating System')
    filePath = "./web/static/os.png"
    if os.path.exists(filePath):
        print("deleting")
        os.remove(filePath)
    plt.savefig(filePath)
    plt.clf()
    return json.dumps({"url": "http://localhost:5000/os.png"})

@app.route("/special_day")
def specialDay():
    plt.hist(data.SpecialDay, bins=20)
    plt.ylabel('Special day')
    filePath = "./web/static/special_day.png"
    if os.path.exists(filePath):
        print("deleting")
        os.remove(filePath)
    plt.savefig(filePath)
    plt.clf()
    return json.dumps({"url": "http://localhost:5000/special_day.png"})

@app.route("/exit_rates")
def exitRates():
    plt.hist(data.ExitRates, bins=20)
    plt.ylabel('Exit Rates')
    filePath = "./web/static/exit.png"
    if os.path.exists(filePath):
        print("deleting")
        os.remove(filePath)
    plt.savefig(filePath)
    plt.clf()
    return json.dumps({"url": "http://localhost:5000/exit.png"})

@app.route("/training_plot")
def trainingPlot():
    data=pd.read_csv('online_shoppers_intention.csv')
    data['OperatingSystems']=data['OperatingSystems'].astype(str)
    data['Browser']=data['Browser'].astype(str)
    data['Region']=data['Region'].astype(str)
    data['TrafficType']=data['TrafficType'].astype(str)
    booleandf = data.select_dtypes(include=[bool])
    booleanDictionary = {True: 'TRUE', False: 'FALSE'}

    for column in booleandf:
        data[column] = data[column].map(booleanDictionary)
    combine = [data]
    opinion_mapping = {"FALSE": 0, "TRUE": 1}
    for dataset in combine:
        dataset['Revenue'] = dataset['Revenue'].map(opinion_mapping)
    y=data['Revenue']
    del data['Revenue']
    data = pd.get_dummies(data)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)  
    clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=1000, alpha=0.0001,
                        solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
    clf.fit(X_train,y_train)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(0.0001))
    plt.plot(clf.loss_curve_)
    filePath = "./web/static/loss.png"
    if os.path.exists(filePath):
        print("deleting")
        os.remove(filePath)
    plt.savefig(filePath)
    y_pred=clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return json.dumps({"url": "http://localhost:5000/loss.png", "score": score})

@app.route('/predict', methods=['POST']) 
def foo():
    if not request.json:
        abort(400)
    submittedData = request.json
    subDF = pd.DataFrame([submittedData])

    data=pd.read_csv('online_shoppers_intention.csv')
    data['OperatingSystems']=data['OperatingSystems'].astype(str)
    subDF['OperatingSystems']=subDF['OperatingSystems'].astype(str)
    data['Browser']=data['Browser'].astype(str)
    subDF['Browser']=subDF['Browser'].astype(str)
    data['Region']=data['Region'].astype(str)
    subDF['Region']=subDF['Region'].astype(str)
    data['TrafficType']=data['TrafficType'].astype(str)
    subDF['TrafficType']=subDF['TrafficType'].astype(str)
    booleandf = data.select_dtypes(include=[bool])
    booleanDictionary = {True: 'TRUE', False: 'FALSE'}
    subDF["Revenue"] = "FALSE"

    for column in booleandf:
        data[column] = data[column].map(booleanDictionary)
        subDF[column] = subDF[column].map(booleanDictionary)
    combine = [data]
    opinion_mapping = {"FALSE": 0, "TRUE": 1}
    for dataset in combine:
        dataset['Revenue'] = dataset['Revenue'].map(opinion_mapping)
    y=data['Revenue']
    del data['Revenue']
    del subDF["Revenue"]
    data.append(subDF, ignore_index=True)
    data = pd.get_dummies(data)
    subDF = data.iloc[-1]
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)  
    clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=1000, alpha=0.0001,
                        solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
    clf.fit(X_train,y_train)
    prediction = clf.predict(subDF.values.reshape(1, -1))
    print(prediction)
    if(prediction[0] == 0):
        return "False"
    else:
        return "True"  




if __name__ == '__main__':
    app.run(debug=True)