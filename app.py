import os
import pickle
import pandas as pd
from flask import Flask, request, render_template
from utils import extract_data

print(os.getcwd())
path = os.getcwd()

with open('Models/DecisionTree', 'rb') as f:
    decisionTree = pickle.load(f)
with open('Models/LogisticRegression', 'rb') as f:
    logisticRegression = pickle.load(f)
with open('Models/NaiveBayes', 'rb') as f:
    naiveBayes = pickle.load(f)


def get_predictions(age, cp, sex, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, req_model, slope, ca, thal):
    data = pd.DataFrame({'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
                         'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
                         'slope': slope, 'ca': ca}, index=[0])
    extract_data(data)
    vals = data.iloc[:].values

    if req_model == 'DecisionTree':
        print(req_model)
        return decisionTree.predict(vals)[0]
    elif req_model == 'LogisiticRegression':
        print(req_model)
        print("get Pred LR")
        return logisticRegression.predict(vals)[0]
    elif req_model == 'NaiveBayes':
        print(req_model)
        return naiveBayes.predict(vals)[0]
    else:
        return "Cannot Predict"


app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    print("REACHED HERE1")
    if request.method == 'POST':
        age = int(request.form.get('age'))
        cp = int(request.form.get('cp'))
        sex = int(request.form.get('sex'))
        trestbps = int(request.form.get('trestbps'))
        chol = int(request.form.get('chol'))
        fbs = int(request.form.get('fbs'))

        restecg = int(request.form.get('restecg'))
        thalach = int(request.form.get('thalach'))
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = int(request.form.get('slope'))
        ca = int(request.form.get('ca'))
        thal = int(request.form.get('thal'))
        req_model = (request.form.get('req_model'))

        target = get_predictions(age, cp, sex, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, req_model, slope,
                                 ca, thal)
        print(target)
        if target == 1:
            sale_making = 'The patient has heart disease'
        else:
            sale_making = 'The patient does not have a heart disease'

        return render_template('home.html', target=target, sale_making=sale_making)
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
