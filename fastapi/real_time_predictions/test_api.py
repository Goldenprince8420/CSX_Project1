import pandas as pd
import requests
import json
import numpy as np
df = pd.read_csv(f'/home/lakshmi/Desktop/lakshmi/Ali/fastapi/test_clean_data.csv')
df = df.drop('Unnamed: 0',axis=1)
header = {"Content-Type": "application/json"}
  
url = 'http://127.0.0.1:8000/score_predict'

payload={"data": df.to_dict(orient="records")}
r = requests.post(url, data=payload)
print(r.json())
