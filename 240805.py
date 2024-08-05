import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
sub_df=pd.read_csv("sample_submission.csv")

house_train = df.copy()
house_test = df_test.copy()
sub = sub_df.copy()

# 숫자형 변수만 선택하기
x = house_train.select_dtypes(include=[int, float])
# 필요없는 열 제거하기
x = x.iloc[:,1:-1]
y = house_train["SalePrice"]
x.isna().sum()

# mode는 최빈값값
#fill_values = {
#    "LotFrontage" : x["LotFrontage"].mean(), # np.float64() 이렇게 나와야함
#    "MasVnrArea" : x["MasVnrArea"].mode()[0], # 시리즈로 나와서 찾아서 가져옴
#    "GarageYrBlt" : x["MasVnrArea"].mean()
# }
x = x.fillna(value=fill_values)

x["LotFrontage"] = x["LotFrontage"].fillna(house_train["LotFrontage"].mean())
x["MasVnrArea"] = x["MasVnrArea"].fillna(house_train["MasVnrArea"].mean())
x["GarageYrBlt"] = x["GarageYrBlt"].fillna(house_train["GarageYrBlt"].mean())

model = LinearRegression()
model.fit(x, y)

slope = model.coef_ 
intercept = model.intercept_ 

test_x = house_test.select_dtypes(include=[int, float])
test_x = test_x.iloc[:,1:]
test_x.isna().sum()

#fill_values = {
#    "LotFrontage" : test_x["LotFrontage"].mean(),
#    "MasVnrArea" : test_x["MasVnrArea"].mean(), 
#    "GarageYrBlt" : test_x["GarageYrBlt"].mean()
#}
#test_x = test_x.fillna(value=fill_values)
test_x = test_x.fillna(test_x.mean()) # 변수별로 mean값을 내서 변수별로 값을 넣어줌

pred_y = model.predict(test_x)
sub["SalePrice"] = pred_y
sub.to_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission10.csv", index=False)
