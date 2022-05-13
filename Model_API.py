import joblib
from flask import Flask,request
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

pipeline = joblib.load("finalized_model.pkl")

@app.route('/predict',methods=['GET', 'POST'])

def predict():
    
    event = json.loads(request.data)
    input_val = pd.DataFrame([event], columns=event.keys())
    pred_val = pd.DataFrame(pipeline.predict(input_val)).values[0][0].round()
    return {"Prediction":pred_val}

if __name__ == '__main__':
    app.run()