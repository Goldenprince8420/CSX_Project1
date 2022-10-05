import pandas as pd
import requests
import json
import numpy as np
df = pd.read_csv(f'/home/lakshmi/Desktop/lakshmi/Ali/fastapi/test_clean_data.csv')
df = df.drop('Unnamed: 0',axis=1)
df_json = df.to_json(orient = "records")
# print(df_json)
# header = {"Content-Type": "application/json"}
  
url = 'http://127.0.0.1:8000/predict'

# payload={"data": df_json}
r = requests.post(url, json=json.loads(df_json))
print(r.json())
