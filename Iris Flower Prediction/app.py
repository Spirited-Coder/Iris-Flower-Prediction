import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#create flask app
app = Flask(__name__)

#load pickle model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
	return render_template("index.html")

@app.route('/predict', methods = ["POST"])
def predict():
	species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
	float_features = [float(x) for x in request.form.values()]
	features = [np.array(float_features)]
	prediction = model.predict(features)

	# Get the predicted class index
	predicted_index = int(prediction[0])
	
	# Use the predicted index to get the flower species from the dictionary
	predicted_species = species_mapping.get(predicted_index, "Unknown")

	return render_template('index.html', prediction_text="The flower species is {}".format(predicted_species))

if __name__=="__main__":
	app.run(debug = True)