# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:34:25 2022

@author: Ganesh Singewar
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict_mpg(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Fuel Efficiency of Vehicle is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)