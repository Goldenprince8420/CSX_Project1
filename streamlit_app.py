import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Home Credit Default Risk')
plt.style.use("ggplot")

# loading the data

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(f'{file}.csv')
    return df

train_df = load_data('train_data_domain_filtered')
result = load_data('results_filtered')

st.subheader('Distribution of Variables')
variables = st.selectbox(
  'Choose Variable for Histogram Plot',
  ('DAYS_BIRTH', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_INCOME_PERCENT', 
  'ANNUITY_INCOME_PERCENT','CREDIT_TERM','DAYS_EMPLOYED_PERCENT'))

fig1 = plt.figure(figsize=(10,8))
train_df[variables]=train_df[variables].abs()
sns.kdeplot(train_df.loc[train_df['TARGET']==0,variables]/365,label='target==0')
sns.kdeplot(train_df.loc[train_df['TARGET']==1,variables]/365,label='target==1')
plt.legend()
plt.xlabel(variables)
plt.ylabel('Density')
plt.title(f'Distribution of {variables} by target value')
st.pyplot(fig1)

st.subheader('Correlation Heatmaps')
dataset = st.selectbox(
  'Choose data for Correlation Heatmap',
  ('Credit_Card_Balance', 'Installments_Payments', 'Previous_Application'))

fig2 = plt.figure(figsize=(20,15))
df = load_data(dataset)
corr = df.corr().abs()
sns.heatmap(corr, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
plt.title('Correlation Heatmap')
plt.savefig(dataset+".png")
st.pyplot(fig2)

# Pie Chart

st.write('Pie Chart')
fig3 = plt.figure(figsize=(8, 8))

labels=['Default', 'Non-Default']
sizes=[result[result['Class']==1]['Class'].count(), result[result['Class']==0]['Class'].count()]

plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', 
        shadow=True,
        startangle=200,
        explode = [0, 0.1])

plt.axis('equal')
plt.savefig("PieChart.png")
st.pyplot(fig3)

# Credit Amount 

st.write('Credit Amount')
amtCredit=result.sort_values(by='AMT_CREDIT', ascending=False)[['SK_ID_CURR', 'AMT_CREDIT']]
amtCredit.set_index('SK_ID_CURR')[:20].plot.barh(figsize=(10, 10))
plt.savefig("CreditAmount.png")
# plt.show()
st.pyplot()


result["Age(years)"] = abs(result['DAYS_BIRTH']) // 365
train_df["Age(years)"] = abs(train_df['DAYS_BIRTH']) // 365

def get_cat(x):
    return str(x//10*10) + '-' + str((x//10*10)+10)

result['Age_cat']=list(map(get_cat, result["Age(years)"]))
train_df["Age_cat"]=list(map(get_cat, train_df["Age(years)"]))

data = result.groupby(by="Age_cat", as_index=False).mean()
data2 = train_df.groupby(by="Age_cat", as_index=False).mean()

# print(data.head())
st.write("Age Vs Amount Income")
fig6 = plt.figure(figsize = (10, 5))
plt.plot(data["Age_cat"], data["AMT_INCOME_TOTAL"])
# result.groupby(['Age(years)','AMT_INCOME_TOTAL']).sum().unstack().plot()
plt.title("Age(years) vs Amount of Income")
plt.savefig("AGE_AMT_OF_INCOME" + ".png")
# plt.show()
st.pyplot(fig6)

st.write("Age vs Total Amount Credit")
fig7 = plt.figure(figsize=(10, 5))
plt.plot(data["Age_cat"], data["AMT_CREDIT"])
plt.title("Age(years) vs Amount of Credit")
plt.savefig("AGE_AMT_OF_CREDIT" + ".png")
# plt.show()
st.pyplot(fig7)
#
st.write("Age vs Flag Own Realty")
fig8 = plt.figure(figsize=(10, 5))
plt.plot(data["Age_cat"], data["FLAG_OWN_REALTY"])
plt.title("Age(years) vs Flag Own Realty")
plt.savefig("AGE_FLAG_OWN_REALITY" + ".png")
# plt.show()
st.pyplot(fig8)
#
st.write("Age vs Days Employed")
fig9 = plt.figure(figsize=(10, 5))
plt.plot(data2["Age_cat"], data2["DAYS_EMPLOYED"])
plt.title("Age(years) vs Days Employed")
plt.savefig("AGE_DAYS_EMPLOYED" + ".png")
plt.show()
st.pyplot(fig9)

# st.write("Years Employeed(Grouped) vs Total Amount Credit")
# fig10 = plt.figure(figsize=(10, 5))
# plt.plot(result.groupby(["Age(years)", "Target"], ), result["AMT_CREDIT"])
# plt.title("Age(years) vs Amount of Income")
# plt.savefig("AGE_AMT_OF_INCOME" + ".png")
# st.pyplot(fig10)
#
# st.write("Years Employeed(Grouped) vs Age Income")


focus={'AMT_CREDIT_PERCENT': "the average between the loan and the income",
       'AMT_APPLICATION':'For how much credit did client ask on the previous application',
       'DAYS_EMPLOYED':'How many days before the application the person started current employment',
       'DAYS_BIRTH':"Client's age in days at the time of application",
       'AMT_GOODS_PRICE':'Goods price of good that client asked for (if applicable) on the previous application',
       'AMT_ANNUITY_x':'Annuity of previous application',
       'AMT_INCOME_TOTAL':'Income of the client',
       'AMT_CREDIT':' Credit amount of the loan'}

st.write(focus)

st.subheader('Result Plots')

try:
    id = st.text_input('Enter Client ID:')
    # id=240007
    prob = result.loc[result['SK_ID_CURR']==id]['TARGET'].values[0]*100
    # print(f'The client {id} has a {str(round(prob, 1))}% risk of defaulting on their loan.')
    st.write(f'The client {id} has a {str(round(prob, 1))}% risk of defaulting on their loan.')
except:
    pass

try:
    result['SK_ID_CURR'] = result['SK_ID_CURR'].astype('str')
    result['DAYS_BIRTH'] = abs(result['DAYS_BIRTH'])
    client = result[result['SK_ID_CURR']==id]
    sameClass = result[result['Class']==int(client['Class'].values[0])]
    if int(client['Class'])==1:
        oppClass=result[result['Class']==0]
    else:
        oppClass=result[result['Class']==1]

    # figs = [None] * len(focus.keys())
    # i = 0

    st.write("Amount of Credit Vs Index")
    key = 'AMT_CREDIT_PERCENT'
    val = focus[key]
    temp = pd.DataFrame(columns=['Target', 'Average', 'SameGroup', 'OppGroup'])
    temp['Target'] = client[key]
    temp['Average'] = np.average(result[key].values)
    temp['SameGroup'] = np.average(sameClass[key].values)
    temp['OppGroup'] = np.average(oppClass[key].values)
    temp = temp.T
    fig100 = plt.figure(figsize=(10, 5))
    plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
    plt.title(key)
    plt.savefig(key + ".png")
    plt.show()
    st.pyplot(fig100)
    #
    # key = 'AMT_APPLICATION'
    # val = focus[key]
    # temp = pd.DataFrame(columns=['Target', 'Average', 'SameGroup', 'OppGroup'])
    # temp['Target'] = client[key]
    # temp['Average'] = np.average(result[key].values)
    # temp['SameGroup'] = np.average(sameClass[key].values)
    # temp['OppGroup'] = np.average(oppClass[key].values)
    # temp = temp.T
    # fig6 = plt.figure(figsize=(10, 5))
    # plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
    # plt.title(key)
    # plt.savefig(key + ".png")
    # plt.show()
    # st.pyplot(fig6)
    # #
    # key = 'DAYS_EMPLOYED'
    # val = focus[key]
    # temp = pd.DataFrame(columns=['Target', 'Average', 'SameGroup', 'OppGroup'])
    # temp['Target'] = client[key]
    # temp['Average'] = np.average(result[key].values)
    # temp['SameGroup'] = np.average(sameClass[key].values)
    # temp['OppGroup'] = np.average(oppClass[key].values)
    # temp = temp.T
    # fig7 = plt.figure(figsize=(10, 5))
    # plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
    # plt.title(key)
    # plt.savefig(key + ".png")
    # plt.show()
    # st.pyplot(fig7)
    #
    # key = 'DAYS_BIRTH'
    # val = focus[key]
    # temp = pd.DataFrame(columns=['Target', 'Average', 'SameGroup', 'OppGroup'])
    # temp['Target'] = client[key]
    # temp['Average'] = np.average(result[key].values)
    # temp['SameGroup'] = np.average(sameClass[key].values)
    # temp['OppGroup'] = np.average(oppClass[key].values)
    # temp = temp.T
    # fig8 = plt.figure(figsize=(10, 5))
    # plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
    # plt.title(key)
    # plt.savefig(key + ".png")
    # plt.show()
    # st.pyplot(fig8)
    #
    # key = 'AMT_GOODS_PRICE'
    # val = focus[key]
    # temp = pd.DataFrame(columns=['Target', 'Average', 'SameGroup', 'OppGroup'])
    # temp['Target'] = client[key]
    # temp['Average'] = np.average(result[key].values)
    # temp['SameGroup'] = np.average(sameClass[key].values)
    # temp['OppGroup'] = np.average(oppClass[key].values)
    # temp = temp.T
    # fig9 = plt.figure(figsize=(10, 5))
    # plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
    # plt.title(key)
    # plt.savefig(key + ".png")
    # plt.show()
    # st.pyplot(fig9)
    #
    # key = 'AMT_ANNUITY_x'
    # val = focus[key]
    # temp = pd.DataFrame(columns=['Target', 'Average', 'SameGroup', 'OppGroup'])
    # temp['Target'] = client[key]
    # temp['Average'] = np.average(result[key].values)
    # temp['SameGroup'] = np.average(sameClass[key].values)
    # temp['OppGroup'] = np.average(oppClass[key].values)
    # temp = temp.T
    # fig10 = plt.figure(figsize=(10, 5))
    # plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
    # plt.title(key)
    # plt.savefig(key + ".png")
    # plt.show()
    # st.pyplot(fig10)
    #
    # key = 'AMT_INCOME_TOTAL'
    # val = focus[key]
    # temp = pd.DataFrame(columns=['Target', 'Average', 'SameGroup', 'OppGroup'])
    # temp['Target'] = client[key]
    # temp['Average'] = np.average(result[key].values)
    # temp['SameGroup'] = np.average(sameClass[key].values)
    # temp['OppGroup'] = np.average(oppClass[key].values)
    # temp = temp.T
    # fig11 = plt.figure(figsize=(10, 5))
    # plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
    # plt.title(key)
    # plt.savefig(key + ".png")
    # plt.show()
    # st.pyplot(fig11)
    #
    # key = 'AMT_CREDIT'
    # val = focus[key]
    # temp = pd.DataFrame(columns=['Target', 'Average', 'SameGroup', 'OppGroup'])
    # temp['Target'] = client[key]
    # temp['Average'] = np.average(result[key].values)
    # temp['SameGroup'] = np.average(sameClass[key].values)
    # temp['OppGroup'] = np.average(oppClass[key].values)
    # temp = temp.T
    # fig12 = plt.figure(figsize=(10, 5))
    # plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
    # plt.title(key)
    # plt.savefig(key + ".png")
    # plt.show()
    # st.pyplot(fig12)

    # for key, val in focus.items():
    #
    #     temp = pd.DataFrame(columns=['Target','Average','SameGroup','OppGroup'])
    #     temp['Target']=client[key]
    #     temp['Average']=np.average(result[key].values)
    #     temp['SameGroup']=np.average(sameClass[key].values)
    #     temp['OppGroup']=np.average(oppClass[key].values)
    #     temp = temp.T
    #     fig5 = plt.figure(figsize=(10, 5))
    #     plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
    #     plt.title(key)
    #     plt.savefig(key+".png")
    #     plt.show()
    #     st.pyplot(fig5)
except:
  print('Please enter client ID again')

st.write('Predicting the credit is default or not')

# Utility function
def yes_no(value):
    if value=="Yes":
       return 1
    else:
       return 0

st.write('Choose the feature values')

own_car = st.selectbox(
  'Choose whether client owned Car or not',
  ("Yes", "No"))
own_car = yes_no(own_car)

own_realty = st.selectbox(
  'Choose whether client owned Realty or not',
  ("Yes", "No"))
own_realty = yes_no(own_realty)

days_birth = st.number_input('Enter the Days of Birth:')

days_employed = st.number_input('Enter the Days of Employed:')

income_amt = st.number_input('Enter the Income Amount:')

credit_amt = st.number_input('Enter the Credit Amount:')

# Create a empty dataframe
df = pd.DataFrame()
# These values comes from web app
df['FLAG_OWN_CAR'] = [own_car]
df['FLAG_OWN_REALTY'] = [own_realty]
df['DAYS_BIRTH'] = [days_birth]
df['DAYS_EMPLOYED'] = [days_employed]
df['AMT_INCOME_TOTAL'] = [income_amt]
df['AMT_CREDIT'] = [credit_amt]

# These values are calculated by taking majority votes
# data['feature'].value_counts().argmax()

df['NAME_CONTRACT_TYPE_x'] = 0
df['CODE_GENDER'] = 0
df['CNT_CHILDREN'] = 0
df['NAME_TYPE_SUITE'] = 0
df['NAME_INCOME_TYPE'] = 0
df['NAME_EDUCATION_TYPE'] = 0
df['NAME_FAMILY_STATUS'] = 0
df['NAME_HOUSING_TYPE'] = 0
df['FLAG_WORK_PHONE'] = 0
df['FLAG_PHONE'] = 0
df['FLAG_EMAIL'] = 0
df['REGION_RATING_CLIENT'] = 0
df['WEEKDAY_APPR_PROCESS_START_x'] = 0
df['HOUR_APPR_PROCESS_START_x'] = 0
df['REG_REGION_NOT_LIVE_REGION'] = 0
df['REG_REGION_NOT_WORK_REGION'] = 0
df['LIVE_REGION_NOT_WORK_REGION'] = 0
df['REG_CITY_NOT_LIVE_CITY'] = 0
df['REG_CITY_NOT_WORK_CITY'] = 0
df['LIVE_CITY_NOT_WORK_CITY'] = 0
df['FLAG_DOCUMENT_3'] = 0
df['FLAG_DOCUMENT_6'] = 0
df['FLAG_DOCUMENT_8'] = 0
df['FLAG_DOCUMENT_11'] = 0
df['FLAG_DOCUMENT_13'] = 0
df['FLAG_DOCUMENT_14'] = 0
df['FLAG_DOCUMENT_15'] = 0
df['FLAG_DOCUMENT_16'] = 0
df['FLAG_DOCUMENT_17'] = 0
df['FLAG_DOCUMENT_18'] = 0

# These values are calculated by taking average
# data['feature'].mean()
df['AMT_ANNUITY_x'] = 29411.8
df['REGION_POPULATION_RELATIVE'] = 0.021226
df['DAYS_REGISTRATION'] = -4967.7
df['DAYS_ID_PUBLISH'] = 3051
df['CNT_FAM_MEMBERS'] = 2.1
df['EXT_SOURCE_2'] = 0.517936
df['EXT_SOURCE_3'] = 0.411173
df['OBS_30_CNT_SOCIAL_CIRCLE'] = 1.4
df['DEF_30_CNT_SOCIAL_CIRCLE'] = 0.1
df['DEF_60_CNT_SOCIAL_CIRCLE'] = 0.1
df['DAYS_LAST_PHONE_CHANGE'] = -1077.8
df['AMT_REQ_CREDIT_BUREAU_DAY'] = 0.0
df['AMT_REQ_CREDIT_BUREAU_WEEK'] = 0.0
df['AMT_REQ_CREDIT_BUREAU_MON'] = 0.0
df['AMT_REQ_CREDIT_BUREAU_QRT'] = 0.5
df['AMT_REQ_CREDIT_BUREAU_YEAR'] = 1.7
df['NAME_CONTRACT_TYPE_y'] = 0.76
df['AMT_ANNUITY_y'] = 15111.41113
df['AMT_APPLICATION'] = 154240.1683
df['AMT_GOODS_PRICE'] = 200458.1548
df['FLAG_LAST_APPL_PER_CONTRACT'] = 1.0
df['NFLAG_LAST_APPL_IN_DAY'] = 1.0
df['NAME_CASH_LOAN_PURPOSE'] = 22.64
df['NAME_CONTRACT_STATUS'] = 0.416
df['DAYS_DECISION'] = -899.8
df['NAME_PAYMENT_TYPE'] = 1.02
df['CODE_REJECT_REASON'] = 6.19
df['NAME_CLIENT_TYPE'] = 1.22
df['NAME_GOODS_CATEGORY'] = 17.66
df['NAME_PORTFOLIO'] = 2.67
df['NAME_PRODUCT_TYPE'] = 0.48
df['SELLERPLACE_AREA'] = 404.48
df['CNT_PAYMENT'] = 14.267221
df['NAME_YIELD_GROUP'] = 1.97
df['AMT_CREDIT_PERCENT'] = 3.167544
df['AMT_APPLICATION_PERCENT'] = 5.477633
df['AMT_GOODS_PRICE_PERCENT'] = 4.367816

st.write(df)

model = pickle.load(open('lgbmodel.pkl', 'rb'))
prediction = model.predict(df)
submit = st.button('Predict Default')

if submit:

  if prediction==1:
    st.write('Home Credit is Default')
  else:
    st.write('Home Credit is not Default')
