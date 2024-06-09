from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/api/v1/predict", methods=["POST"])
def predict():
    return predict(request)


@app.route("/api/v1/trainning", methods=["GET"])
def trainning():
    return trainning()
