import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 직선의 방정식
# y = 2x+3의 그래프를 그려보세요!
a = 1
b = 2

x = np.linspace(-5, 5, 100)
y = a*x + b

plt.plot(x, y)
plt.axvline(0, color="black")
plt.axhline(0, color="black")
plt.show()
plt.clf()

#===================================================
# 기울기는 x가 하나 증가하면 y가 얼마나 증가하는지 알 수 있음
a = 80
b = 5

x = np.linspace(0, 5, 100)
y = a*x + b

house_train = pd.read_csv('train.csv')
house_train=house_train[["Id", "BedroomAbvGr", "SalePrice"]]
my_df = house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"] = my_df["SalePrice"]/1000
plt.scatter(x=my_df["BedroomAbvGr"], y=my_df["SalePrice"])
plt.plot(x, y, color="blue")
plt.show()
plt.clf()

#==================================================

house_test = pd.read_csv('test.csv')
a=46
b=63
sub_df=pd.read_csv("sample_submission.csv")
sub_df["SalePrice"] = (a * house_test["BedroomAbvGr"] + b) * 100
sub_df.to_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission4.csv", index=False)

# 직선 성능 평가
# 원래 가격 알아야 해서 train.csv 가져와야 함
a=46
b=63

# y_hat 어떻게 구할까?
y_hat = (a * house_train["BedroomAbvGr"] + b) * 1000
# y는 어디에 있는가?
y = house_train["SalePrice"]

np.abs(y - y_hat) # 절대거리
np.sum(np.abs(y-y_hat)) # 절대거리의 합(거리)이 작으면 작을수록 성능이 좋음 # 절댓값
np.sum((y-y_hat)**2) # 절대거리의 합(거리)이 작으면 작을수록 성능이 좋음 # 제곱

#=================================================
# !pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 회귀분석 적합(fit)하기
# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
slope = model.coef_[0] # 기울기 a (0이 없으면 array로 나온다! 그러니 수를 사용하므로 [0]을 이용)
intercept = model.intercept_ # y절편 b
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x) # 직선 식에 x 값 넣기 -> 직선의 y 값 도출

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='LinearRegression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

#=============================================

house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
sub_df=pd.read_csv("sample_submission.csv")

x = np.array(house_train["BedroomAbvGr"]).reshape(-1, 1) # np.array에만 reshape이 적용됨(2차원 array)
y = np.array(house_train["SalePrice"])/1000

model = LinearRegression()

# 모델 학습(a,b를 찾으라)
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌(minimize 해줌)

# 회귀 직선의 기울기와 절편
slope = model.coef_[0] # 기울기 a: np.float64(16.38101698)
intercept = model.intercept_ # y절편 b: np.float64(133.96602049739172)
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x) # 직선 식에 x 값 넣기 -> 직선의 y 값 도출

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='LinearRegression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

#=========================================

test_x = np.array(house_test["BedroomAbvGr"]).reshape(-1, 1)
pred_y = model.predict(test_x) # test 셋에 대한 집값, predict이 알아서 계산

sub_df=pd.read_csv("sample_submission.csv")
sub_df["SalePrice"] = pred_y * 1000
sub_df.to_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission4.csv", index=False)
