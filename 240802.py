import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 함수 y=x^2 +3 최소값이 나오는 입력값 구하기
def my_f(x):
    return x**2+3

my_f(3)

# 초기 추정값
initial_guess = [0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# z = x^2 + y^2 +3 최소값이 나오는 입력값 구하기
# 변수가 2개일 때 x라 해놓고, x 첫번째 값은 x, 두번째 값은 y
def my_f2(x):
    return x[0]**2+x[1]**2+3

my_f2([1,3])

# 초기 추정값
initial_guess = [1, 3]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# f(x, y, z) = (x-1)^2 + (y-2)^2 +(z-4)^ + 7 최소값과 최소값이 되는 자리
def my_f3(x):
    return ((x[0]-1)**2)+((x[1]-2)**2)+((x[2]-4)**2)+7

my_f3([3, 6, 5])

# 초기 추정값
initial_guess = [1, 2, 4]

# 최소값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

#=======================================
# 최소값을 찾을 다변수 함수 정의
def my_f(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 회귀직선 구하기

import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat)))

line_perform([36, 68])

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

#===========================================
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
sub_df=pd.read_csv("sample_submission.csv")

# 이상치 탐색 및 이상치 제거
house_train = house_train.query("GrLivArea <= 4500")

x = np.array(house_train["GrLivArea"]).reshape(-1, 1)
y = house_train["SalePrice"]/1000

model = LinearRegression()
model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_ 

y_pred = model.predict(x) 

plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='LinearRegression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

test_x = np.array(house_test["GrLivArea"]).reshape(-1, 1)
pred_y = model.predict(test_x)

sub_df=pd.read_csv("sample_submission.csv")
sub_df["SalePrice"]=pred_y * 1000
sub_df.to_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission5.csv", index=False)

#==================================================
# 변수 2개 사용하기

house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
sub_df=pd.read_csv("sample_submission.csv")

house_train = house_train.query("GrLivArea <= 4500")

# x = np.array(house_train[["GrLivArea", "GarageArea"]]).reshape(-1, 2) # 세로로 만들어주기 위해 -1을 한 거임
x = house_train[["GrLivArea" ,"GarageArea"]]
# x = house_train[["GrLivArea"]] # 판다스 프레임(열이 생김, size 표현해줌)
# x = house_train["GrLivArea"] # 판다스 시리즈(데이터 타입과 기링만 나옴), 차원 없음
y = house_train["SalePrice"]

model = LinearRegression()
model.fit(x, y)

slope = model.coef_ 
intercept = model.intercept_ 

def my_houseprice(x,y):
    return model.coef_[0]*x + model.coef_[1]*y + model.intercept_

my_houseprice(300, 55)
temp_result = my_houseprice(house_test["GrLivArea"],house_test["GarageArea"])

test_x = house_test[["GrLivArea", "GarageArea"]]
pred_y = model.predict(test_x)

# 결측치 확인
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

# 그래프 당연히 안 그려짐
y_pred = model.predict(x) 

plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='LinearRegression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

#========================================
# 3차원
# 회기 평면
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
sub_df=pd.read_csv("sample_submission.csv")

house_train = house_train.query("GrLivArea <= 4500")

x = house_train[["GrLivArea" ,"GarageArea"]]
y = house_train["SalePrice"]

model = LinearRegression()
model.fit(x, y)

slope_grlivarea = model.coef_[0]
slope_garagearea = model.coef_[1]
intercept = model.intercept_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트
ax.scatter(x['GrLivArea'], x['GarageArea'], y, color='blue', label='Data points')

# 회귀 평면
GrLivArea_vals = np.linspace(x['GrLivArea'].min(), x['GrLivArea'].max(), 100)
GarageArea_vals = np.linspace(x['GarageArea'].min(), x['GarageArea'].max(), 100)
GrLivArea_vals, GarageArea_vals = np.meshgrid(GrLivArea_vals, GarageArea_vals)
SalePrice_vals = intercept + slope_grlivarea * GrLivArea_vals + slope_garagearea * GarageArea_vals

ax.plot_surface(GrLivArea_vals, GarageArea_vals, SalePrice_vals, color='red', alpha=0.5)

# 축 라벨
ax.set_xlabel('GrLivArea')
ax.set_ylabel('GarageArea')
ax.set_zlabel('SalePrice')

plt.legend()
plt.show()
plt.clf()
