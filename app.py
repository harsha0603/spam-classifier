import os
import sys
from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get('text')
        
        # Create a DataFrame with the input data
        data = pd.DataFrame({'text': [text]})
        
        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        
        try:
            # Make predictions
            predictions = predict_pipeline.predict(data)
            
            # Determine the prediction result
            prediction_result = 'Spam' if predictions[0] == 1 else 'Not Spam'
        
        except CustomException as e:
            prediction_result = f"An error occurred: {e}"
        
        # Return the prediction result on the same page
        return render_template('index.html', prediction=prediction_result)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
