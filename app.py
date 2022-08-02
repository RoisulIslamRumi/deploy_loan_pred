# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:48:40 2022

@author: roisul
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('log_reg.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # To render results on html gui
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)[0]
    
    if prediction == 1:
        pred_text = "Applicant is approved for loan."
    else:
        pred_text = "Applicant is rejected for loan."
    
    return render_template('index.html', prediction_text=pred_text)

@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)