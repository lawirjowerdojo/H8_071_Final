import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)
model = pickle.load(open('model\survivors.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature=[]
    for val in request.form.values():
        feature.append(float(val))
    feature=np.array([feature]).T.reshape(1, -1)
    outcome = {
        0: 'Did not survive',
        1: 'Survived'
    }
    prediction = f'Prediction: {outcome[model.predict(feature)[0]]}'
    return render_template('main.html', prediction_text=prediction)

# @app.route('/predict/results',methods=['GET'])
# def predictionresult():
#     outcome = {
#         0: 'Did not survive',
#         1: 'Survived'
#     }
#     prediction = outcome[model.predict(feature)[0]]    
#     return render_template('result.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)