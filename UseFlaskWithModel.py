# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import string
import re

# instantiate flask 
app = flask.Flask(__name__)

# load the model, and pass in the custom metric function

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

model = load_model('C:\\Development\\ML\\VS\\TestSentimentV2')

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x = pd.DataFrame.from_dict(params, orient='index').transpose()
       
    data["prediction"] = str(model.predict(x)[0][0])
    data["success"] = True

    # return a response in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(host='0.0.0.0')