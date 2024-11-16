from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from model.train_model import load_data, prepare_data, train_naive_bayes
from model.crop_model import predict_crop

app = Flask(__name__)

# Load and prepare data
data = load_data('crop_recommendation.csv')
X, y, label_encoder = prepare_data(data)

# Train the model
model = train_naive_bayes(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Prepare input data for prediction
    field_conditions = [N, P, K, temperature, humidity, ph, rainfall]
    
    # Make prediction
    recommended_crop = predict_crop(model, field_conditions, label_encoder)
    
    return render_template('index.html', prediction=recommended_crop)

if __name__ == '__main__':
    app.run(debug=True)