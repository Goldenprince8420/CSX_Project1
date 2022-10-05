from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import lightgbm as lgb
import re

train_data = pd.read_csv('/home/lakshmi/Desktop/lakshmi/Ali/fastapi/new_train_clean_data.csv')

df = train_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

x = df.drop(['TARGET','Unnamed0'],axis=1).copy()
#print(x.columns)
y = df['TARGET'].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)

# lgb = lgb.LGBMClassifier(class_weight='balanced',n_estimators=300,learning_rate=0.05)
# lgb.fit(x_train, y_train)

test = pd.read_csv('/home/lakshmi/Desktop/lakshmi/Ali/fastapi/train_clean_data.csv')
test = test.drop('Unnamed: 0',axis=1)
# print(test.shape)
print(test.dtypes[40:75])
# predict = lgb.predict(test)
# print(predict)
# pickle.dump(lgb, open('lgbmodel.pkl', 'wb'))
# print(x.shape)

model = pickle.load(open('lgbmodel.pkl', 'rb'))
predict = model.predict(x_test)
# print(predict)