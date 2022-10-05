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
    # contract_type = data['NAME_CONTRACT_TYPE_x']
    # # print(contract_type)
    # gender = data['CODE_GENDER']
    # own_car = data['FLAG_OWN_CAR']
    # own_reality = data['FLAG_OWN_REALTY']
    # children = data['CNT_CHILDREN']
    # total_income = data['AMT_INCOME_TOTAL']
    # credit = data['AMT_CREDIT']
    # annuity = data['AMT_ANNUITY_x']
    # suite_type = data['NAME_TYPE_SUITE']
    # income_type = data['NAME_INCOME_TYPE']
    # education_type = data['NAME_EDUCATION_TYPE']
    # family_status = data['NAME_FAMILY_STATUS']
    # housing_type = data['NAME_HOUSING_TYPE']
    # population_relative = data['REGION_POPULATION_RELATIVE']
    # birth = data['DAYS_BIRTH']
    # employed = data['DAYS_EMPLOYED']
    # registration = data['DAYS_REGISTRATION']
    # id_publish = data['DAYS_ID_PUBLISH']
    # work_phone = data['FLAG_WORK_PHONE']
    # phone = data['FLAG_PHONE']
    # email = data['FLAG_EMAIL']
    # fam_members = data['CNT_FAM_MEMBERS']
    # rating_client = data['REGION_RATING_CLIENT']
    # week_process_start = data['WEEKDAY_APPR_PROCESS_START_x']
    # hour_process_start = data['HOUR_APPR_PROCESS_START_x']
    # reg_region_not_live = data['REG_REGION_NOT_LIVE_REGION']
    # reg_region_not_work = data['REG_REGION_NOT_WORK_REGION']
    # live_region_not_work = data['LIVE_REGION_NOT_WORK_REGION']
    # reg_city_not_live = data['REG_CITY_NOT_LIVE_CITY']
    # reg_city_not_work = data['REG_CITY_NOT_WORK_CITY']
    # live_city_not_work = data['LIVE_CITY_NOT_WORK_CITY']
    # ext_source_2 = data['EXT_SOURCE_2']
    # ext_source_3 = data['EXT_SOURCE_3']
    # obs_30_cnt = data['OBS_30_CNT_SOCIAL_CIRCLE']
    # def_30_cnt = data['DEF_30_CNT_SOCIAL_CIRCLE']
    # def_60_cnt = data['DEF_60_CNT_SOCIAL_CIRCLE']
    # last_phone_change = data['DAYS_LAST_PHONE_CHANGE']
    # flag_document_3 = data['FLAG_DOCUMENT_3']
    # flag_document_6 = data['FLAG_DOCUMENT_6']
    # flag_document_8 = data['FLAG_DOCUMENT_8']
    # flag_document_11 = data['FLAG_DOCUMENT_11']
    # flag_document_13 = data['FLAG_DOCUMENT_13']
    # flag_document_14 = data['FLAG_DOCUMENT_14']
    # flag_document_15 = data['FLAG_DOCUMENT_15']
    # flag_document_16 = data['FLAG_DOCUMENT_16']
    # flag_document_17 = data['FLAG_DOCUMENT_17']
    # flag_document_18 = data['FLAG_DOCUMENT_18']
    # bureau_day = data['AMT_REQ_CREDIT_BUREAU_DAY']
    # bureau_week = data['AMT_REQ_CREDIT_BUREAU_WEEK']
    # bureau_mon = data['AMT_REQ_CREDIT_BUREAU_MON']
    # bureau_qrt = data['AMT_REQ_CREDIT_BUREAU_QRT']
    # bureau_year = data['AMT_REQ_CREDIT_BUREAU_YEAR']
    # contract_type_y = data['NAME_CONTRACT_TYPE_y']
    # annuity_y = data['AMT_ANNUITY_y']
    # application = data['AMT_APPLICATION']
    # goods_price = data['AMT_GOODS_PRICE']
    # last_appl_per_contract = data['FLAG_LAST_APPL_PER_CONTRACT']
    # last_appl_in_day = data['NFLAG_LAST_APPL_IN_DAY']
    # cash_loan_purpose = data['NAME_CASH_LOAN_PURPOSE']
    # contract_status = data['NAME_CONTRACT_STATUS']
    # decision = data['DAYS_DECISION']
    # payment_type = data['NAME_PAYMENT_TYPE']
    # reject_reason = data['CODE_REJECT_REASON']
    # client_type = data['NAME_CLIENT_TYPE']
    # goods_category = data['NAME_GOODS_CATEGORY']
    # portfolio = data['NAME_PORTFOLIO']
    # product_type = data['NAME_PRODUCT_TYPE']
    # seller_place_area = data['SELLERPLACE_AREA']
    # cnt_payment = data['CNT_PAYMENT']
    # yield_group = data['NAME_YIELD_GROUP']
    # credit_percent = data['AMT_CREDIT_PERCENT']
    # application_percent = data['AMT_APPLICATION_PERCENT']
    # good_price_percent = data['AMT_GOODS_PRICE_PERCENT']

    # features_list = [[contract_type, gender, own_car, own_reality, children, total_income, credit, annuity,
    # suite_type,income_type,education_type,family_status,housing_type,population_relative,
    # birth,employed,registration,id_publish,work_phone,phone,email,fam_members,rating_client,week_process_start,
    # hour_process_start,reg_region_not_live,reg_region_not_work,live_region_not_work,reg_city_not_live,
    # reg_city_not_work,live_city_not_work,ext_source_2,ext_source_3,obs_30_cnt,def_30_cnt,def_60_cnt,
    # last_phone_change,flag_document_3,flag_document_6,flag_document_8,flag_document_11,flag_document_13,flag_document_14,
    # flag_document_15,flag_document_16,flag_document_17,flag_document_18,bureau_day,bureau_week,bureau_mon,bureau_qrt,bureau_year,
    # contract_type_y,annuity_y,application,goods_price,last_appl_per_contract,last_appl_in_day,cash_loan_purpose,contract_status,
    # decision,payment_type,reject_reason,client_type,goods_category,portfolio,product_type,seller_place_area,
    # cnt_payment,yield_group,credit_percent,application_percent,good_price_percent]]
    # print(len(features_list))
    # # print(prediction_list)
    # features = np.array(features_list)
    # features = features.reshape(1,-1)
    df = pd.DataFrame([data])
    print(df.shape)
    prediction = model.predict(df)
    # print(str(prediction[0]))
    print(prediction[0])
    return {'prediction' : prediction[0]}


if __name__ == '__main__':
   uvicorn.run(app,host='127.0.0.1', port=8000)


