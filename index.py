from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session
import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_bootstrap import Bootstrap

sess = tf.Session()
global graph
graph = tf.get_default_graph() 

set_session(sess)

model = load_model("model/chest_xray_model.h5")

def predict_image(filename):
	dsize = (224, 224)
	image = cv2.imread("static/predictions/"+filename)
	image = cv2.resize(image, dsize)

	image = np.expand_dims(image, axis=0)

	print(filename)
	print(image.shape)

	global graph

	with graph.as_default():
		set_session(sess)
		predictions = model.predict(image)[0]

		print(predictions)

		return predictions

app = Flask(__name__)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True

Bootstrap(app)

app.secret_key = 'flask_intelligent_app'

photos = UploadSet("photos", IMAGES)

app.config["UPLOADED_PHOTOS_DEST"] = "static/predictions"
configure_uploads(app, photos)

filename = ""

@app.route("/")
def home():
	return render_template("index.html", image_name=filename)

@app.route("/predict", methods=["GET", 'POST'])
def predict():
	if request.method == "POST" and 'photo' in request.files:
		filename = photos.save(request.files['photo'])

		predicted_value = predict_image(filename)[0]

		if predicted_value >= 0.5:
			predicted_value = "PNEUMONIA " + str(predicted_value)
		else:
			predicted_value = "NORMAL " + str(predicted_value)

		return jsonify(prediction=predicted_value)

	return render_template("new_html.html")

if __name__ == "__main__":
	app.run(debug=True)