from email.mime import application
from parser import suite
from typing import Union
import pandas as pd
import uvicorn
from fastapi import FastAPI
# from fastapi.encoders import jsonable_encoder
from schema import Credit_Score
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open('lgbmodel.pkl', 'rb'))

@app.post("/score_predict")
def predict_credit_score(data: Credit_Score):
    data = data.dict()
    # print(data)
    df = pd.DataFrame([data])
    pred = model.predict(df)
    pred_prob = model.predict_proba(df)
    return {'prediction' : pred[0].tolist(),'prediction_probabilities' : pred_prob[0].max().tolist()}


if __name__ == '__main__':
   uvicorn.run(app,host='127.0.0.1', port=8000)


