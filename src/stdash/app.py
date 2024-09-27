import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import datetime as dt
import numpy as np

st.title("CNN JOB MON")

def load_data():
    url = 'http://43.202.66.118:8077/all'
    r = requests.get(url)
    d = r.json()

    return d
    
data = load_data()
df = pd.DataFrame(data)
df

# request_time을 datetime으로 변환
df['request_time'] = pd.to_datetime(df['request_time'])

# request_time을 시간 단위로 그룹화하고 요청 수 계산
df['request_hour'] = df['request_time'].dt.strftime('%Y-%m-%d %H')  # 시간 단위로 포맷
df_grouped = df.groupby('request_hour').size().reset_index(name='Number of Requests')

# x축을 시간으로, y축을 요청 수로 하는 그래프 그리기
x = df_grouped['request_hour']
y = df_grouped['Number of Requests']

plt.figure(figsize=(10, 6))
plt.bar(x, y)
plt.plot(x,y, 'r')
plt.xticks(rotation=45)  # x축 레이블이 겹치지 않도록 90도 회전
plt.xlabel('Date and Time')
plt.ylabel('Number of Requests')
plt.title('Requests by Date and Time')

plt.tight_layout()  # 레이아웃 조정
plt.show()

st.pyplot(plt)


df['request_user']=df['request_user'].astype(str)
requestu=df.groupby('request_user').size()

# df_groupuser['request_count'] = df.groupby('request_user').size['num']
# #df_groupuser['key'] = df_grouped.index
# #df_groupuser['key']
# df_groupuser['request_count']

df['prediction_model']=df['prediction_model'].astype(str)
predictionm=df.groupby('prediction_model').size()

# #size()
# df_prediction = df.groupby('prediction_model').size()
# df_prediction

plt.figure(figsize=(16, 8))
plt.bar(requestu.index, requestu.values, width=0.4, label='request', color='blue', align='center')
plt.bar(predictionm.index, predictionm.values, width=0.4, label='predicton', color='red', align="edge")

plt.title('Request user(BLUE) and Prediction(RED) Counts over Time')
plt.xlabel('time')
plt.ylabel('count')

# plt.figure(figsize=(16,8))
# #.index .values
# plt.bar(df_groupuser.index, df_groupuser.values)
# plt.bar(df_prediction.index, df_prediction.values)

st.pyplot(plt)

