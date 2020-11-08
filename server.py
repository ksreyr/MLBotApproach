import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from flask import jsonify
from utils import Utils
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from sklearn.model_selection import train_test_split
import csv

app = Flask(__name__)


datosPR={}
Uid='nonekey'
key='none'

# routes
@app.route('/')
def Index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    utils = Utils()
   
    return jsonify({'message': 'go to home'})

@app.route('/api', methods=['POST'])
def api():
    utils = Utils()
    if request.method == 'POST':
        wordsU = request.form['words']
        Uid = request.form['Uid'] # datos de usuario
        key = request.form['key']  
        if utils.confirmKey(key) == True :
            x_test = utils.vectorized_fiting('./in/ropa.csv',utils.textData_cleaning(wordsU))
            prediction = model.predict(x_test)
            datosPR[wordsU]=prediction[0] #aqui guardan las preguntas y respuestas
            archivo = csv.writer(open('in/'+Uid+'.csv',"a", newline=''))#ab
            archivo.writerow([wordsU,prediction[0]])
            return jsonify({'prediccion': list(prediction)})
        else:
            return jsonify({'messages': 'debes tener una key de api valida'})

# TODO Sebestian Cristian Jimmy ampliar los request para un post con mensajes
if __name__ == "__main__":
    model = joblib.load('./models/0.35873440285204994')
    app.run(port=8080, debug=True)
