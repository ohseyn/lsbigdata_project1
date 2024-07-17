import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## 데이터 전처리

# 데이터 불러오기
df = pd.read_csv("C:/Doit_Python-main/Data/발화요인에_대한_월별_화재발생현황.csv")
df
df.columns

# 연도별로 나눔
data_2020 = df[['항목'] + df.filter(like='2020').columns.tolist()]
data_2020

data_2021 = df[['항목'] + df.filter(like='2021').columns.tolist()]
data_2021

data_2022 = df[['항목'] + df.filter(like='2022').columns.tolist()]
data_2022

# 제품결함 제거
data_2022 = data_2022.drop(columns = "2022.11")
data_2022.columns

# 0번째 행을 열로 가져오기
data_2020.columns = data_2020.iloc[0] 
data_2020
data_2021.columns = data_2021.iloc[0] 
data_2021
data_2022.columns = data_2022.iloc[0] 
data_2022

# 합계부터 내용 가져오기
data_2020 = data_2020[2:]
data_2021 = data_2021[2:]
data_2022 = data_2022[2:]

# 인덱스 재정렬(리셋)
data_2020 = data_2020.reset_index(drop=True)
data_2021 = data_2021.reset_index(drop=True)
data_2022 = data_2022.reset_index(drop=True)

# 변환할 열 목록
columns_to_convert = ['계', '전기적요인', '기계적요인', '화학적요인', '가스누출', 
                      '교통사고', '부주의', '기타', '자연적요인', '방화', '방화의심', '미상']

# 각 열에 대해 pd.to_numeric 적용
for column in columns_to_convert:
    data_2020[column] = pd.to_numeric(data_2020[column])
    data_2021[column] = pd.to_numeric(data_2021[column])
    data_2022[column] = pd.to_numeric(data_2022[column])
    
data_2020.info()
data_2021.info()
data_2022.info()

# 파생 변수 만들기
data_2020["계절"] = np.where(data_2020["항목"].isin(["3월", "4월", "5월"]), "봄", 
                    np.where(data_2020["항목"].isin(["6월", "7월", "8월"]), "여름",
                    np.where(data_2020["항목"].isin(["9월", "10월", "11월"]), "가을", "겨울")))
                    
data_2021["계절"] = np.where(data_2021["항목"].isin(["3월", "4월", "5월"]), "봄", 
                    np.where(data_2021["항목"].isin(["6월", "7월", "8월"]), "여름",
                    np.where(data_2021["항목"].isin(["9월", "10월", "11월"]), "가을", "겨울")))
                    
data_2022["계절"] = np.where(data_2022["항목"].isin(["3월", "4월", "5월"]), "봄", 
                    np.where(data_2022["항목"].isin(["6월", "7월", "8월"]), "여름",
                    np.where(data_2022["항목"].isin(["9월", "10월", "11월"]), "가을", "겨울")))
                    
season20 = data_2020.groupby('계절').agg(계절별화재=('계','sum'))
season21 = data_2021.groupby('계절').agg(계절별화재=('계','sum'))
season22 = data_2022.groupby('계절').agg(계절별화재=('계','sum'))

# 그래프

