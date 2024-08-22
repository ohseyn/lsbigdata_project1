import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

# 다항회귀분석
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

# mode는 최빈값
#fill_values = {
#    "LotFrontage" : x["LotFrontage"].mean(), # np.float64() 이렇게 나와야함
#    "MasVnrArea" : x["MasVnrArea"].mode()[0], # 시리즈로 나와서 찾아서 가져옴
#    "GarageYrBlt" : x["MasVnrArea"].mean()
#    }
#x = x.fillna(value=fill_values)

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

#===============================================
from scipy.stats import norm

# 직선의 방정식
# y = 2x+3의 그래프를 그려보세요!
a = 2
b = 3

x = np.linspace(0, 100, 400)
y = a*x + b

# np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100), 20)
epsilon_i= norm.rvs(loc=0, scale=10, size=20) # 분산이 크면 오차가 커짐(신뢰도 저하)
obs_y = 2*obs_x + 3 + epsilon_i

model = LinearRegression()
obs_x=obs_x.reshape(-1, 1)
model.fit(obs_x, obs_y)

slope = model.coef_[0]
intercept = model.intercept_

s_x = np.linspace(0, 100, 400)
i_y = model.coef_[0] * x + model.intercept_

plt.plot(s_x, i_y, color='red', label='LinearRegression')
plt.plot(x, y, color="black")
plt.scatter(obs_x, obs_y, color="blue", s=3)
plt.xlim([0, 100])
plt.ylim([0, 300])
plt.show()
plt.clf()

# !pip install statsmodels
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary()) # coef 열 -> const: b(절편) x1: a(기울기) (model 값과 동일)
