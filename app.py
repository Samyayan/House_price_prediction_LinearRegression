import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


with open("house_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["HEAD" , "POST"])
def predict():
    int_features= [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction= model.predict(final_features)

    return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))


if __name__=="__main__":
    app.run(debug=True)

