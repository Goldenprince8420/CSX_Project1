from typing import Union
import pandas as pd
import uvicorn
from fastapi import FastAPI
# from fastapi.encoders import jsonable_encoder
from schema import Credit_Score_In
import pickle
import numpy as np
from typing import List

app = FastAPI()

model = pickle.load(open('lgbmodel.pkl', 'rb'))

@app.post("/predict")
def predict_credit_score(features: List[Credit_Score_In]):
    df = pd.DataFrame([i.dict() for i in features])
    pred = model.predict(df)
    # print(len(pred))
    pred_prob = model.predict_proba(df)
    # print(len(pred_prob))

    results = pd.DataFrame()

    for i in range(len(pred)):
        dct = {'prediction' : pred[i],'pred_proba' : pred_prob[i].max()}
        df_dct = pd.DataFrame([dct])
        results = pd.concat([results, df_dct], ignore_index=True)
        print(results.head())
      
    return results.to_dict(orient="records")

if __name__ == '__main__':
   uvicorn.run(app,host='127.0.0.1', port=8000)
