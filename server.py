import joblib
import numpy as np
from flask import Flask, render_template, request,redirect,url_for,jsonify,make_response
from flask import jsonify

app = Flask(__name__)

# routes

@app.route('/')
def Index():
	return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([7.594444821, 7.479555538, 1.616463184, 1.53352356,
                       0.796666503, 0.635422587, 0.362012237, 0.315963835, 2.277026653])
    prediction = model.predict(X_test.reshape(1, -1))
    return jsonify({'prediccion': list(prediction)})

@app.route('/api', methods=['POST'])
def api():
	if request.method == 'POST':
		
		wordsU = request.form['words']
		#key = request.form['key']
		print(wordsU);

	return jsonify({'prediccion': wordsU})

# TODO Sebestian Cristian Jimmy ampliar los request para un post con mensajes
if __name__ == "__main__":
    model = joblib.load('./models/0.9319120119763165')
    app.run(port=8080, debug=True)


