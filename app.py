from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    features = [
        float(request.form['overall_qual']),
        float(request.form['gr_liv_area']),
        float(request.form['bedrooms']),
        float(request.form['full_bath']),
        float(request.form['tot_rms']),
        float(request.form['year_built']),
        float(request.form['lot_area'])
    ]
    final_features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted House Price: ${output}')

if __name__ == '__main__':
    app.run(debug=True)