import numpy as np

# 가로 벡터 * 세로 벡터
a = np.arange(1,4)
b = np.array([3, 6, 9]) # 3*x
a.dot(b)

# 행렬 * 벡터
a = np.array([1, 2, 3, 4]).reshape((2,2), order="F")
b = np.array([5, 6]).reshape(2,1)
a.dot(b) # a @ b

# 행렬 * 행렬
a = np.array([1, 2, 3, 4]).reshape((2,2), order="F")
b = np.array([5, 6, 7, 8]).reshape((2,2), order="F")
a @ b

# 행렬 * 행렬 예시1
a = np.array([1, 2, 1, 0, 2, 3]).reshape((2,3))
b = np.array([1, 0, -1, 1 ,2, 3]).reshape((3,2))
a @ b

# 행렬 * 행렬 예시2
a = np.array([3, 5, 7, 2, 4, 9, 3, 1, 0]).reshape((3, 3))
np.eye(3)

a @ np.eye(3)
np.eye(3) @ a

# 행렬 뒤집기(transpose)
a.transpose()
b = a[:, 0:2] # 행 다 가져오고, 열은 0,1 행만
b.transpose()

#=========================================
# 회귀분석 예측식
x=np.array([13, 15,
            12, 14,
            10, 11,
            5, 6]).reshape(4,2)

vec1=np.repeat(1,4).reshape(4,1)
matX=np.hstack((vec1, x)) # 옆으로 붙이기

beta_vec=np.array([2, 3, 1]).reshape(3,1)
# beta_vec=np.array([2, 0, 1]).reshape(3,1)
# beta_vec=np.array([2, -1, 1]).reshape(3,1)
matX @ beta_vec

y=np.array([20, 19, 20, 12]).reshape(4,1)
(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec)

#=========================================
# 2*2 역행렬
a = np.array([1, 5, 3, 4]).reshape(2, 2)
a_inv = (-1/11)*np.array([4, -5, -3, 1]).reshape(2,2)
a @ a_inv

# 3*3 역행렬
a = np.array([-4, -6, 2,
              5, -1, 3,
              -2, 4, -3]).reshape(3,3)
a_inv = np.linalg.inv(a)
np.round(a @ a_inv, 3)

# 역행렬 존재하지 않는 경우(선형종속)
b = np.array([1, 2, 3,
              2, 4, 5,
              3, 6, 7]).reshape(3,3)
b_inv = np.linalg.inv(b) # 에러남

# 역행렬이 있는지 없는지 확인하기 위해 쓰는 함수
np.linalg.det(b) # 0이면 역행렬이 존재하지 않음
np.linalg.det(a) # 0이 아니면 역행렬이 존재

#======================================
# 벡터 형태로 beta 구하기
XtX_inv = np.linalg.inv((matX.transpose() @ matX))
Xty = matX.transpose() @ y
beta_hat = XtX_inv @ Xty

# model.fit()으로 beta 구하기
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(matX[:, 1:], y)
model.coef_
model.intercept_
beta_hat

#==========================================
# minimize로 beta 구하기(손실 함수)
from scipy.optimize import minimize

# (y-X*beta_hat)T(y-X*beta_hat): 절대값 -> 함수화
# a: (y-X*beta_hat)
def line_perform(beta):
    beta = np.array(beta).reshape(3, 1)
    a = (y - matX @ beta)
    return (a.transpose() @ a)

# 들어가는 숫자는 beta0, beta1, beta2
line_perform([6, 1, 3]) # 임의로 넣음
# 최솟값 나오는 beta 값(람다가 0)
line_perform([8.55768897, 5.961567, -4.38463997])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)
result.fun # 최소값
result.x # beta0, beta1, beta2

#====================================
# minimize로 Lasso beta 구하기(패널티 있는 손실함수)
from scipy.optimize import minimize

# (y-X*beta_hat)T(y-X*beta_hat): 절대값(정사각형 넓이) -> 함수화
# a: (y-X*beta_hat) = (y-y_hat)
def line_perform_lasso(beta):
    beta = np.array(beta).reshape(3, 1)
    a = (y - matX @ beta) # (n,1)
    return (a.transpose() @ a) + 3*np.abs(beta).sum()

# 들어가는 숫자는 beta0, beta1, beta2
# 최솟값 나오는 beta 값
line_perform_lasso([3.76, 1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)
result.fun # 최소값
result.x # beta0, beta1, beta2

#========================================
# minimize로 Lasso beta 구하기(패널티 있는 손실함수)
from scipy.optimize import minimize

# (y-X*beta_hat)T(y-X*beta_hat): 절대값 -> 함수화
# a: (y-X*beta_hat)
def line_perform_lasso(beta):
    beta = np.array(beta).reshape(3, 1)
    a = (y - matX @ beta)
    # 1부터 가져온 이유는 람다 숫자가 클 때
    # beta0 값에 있음 숫자가 커지므로 우리는 숫자를 작게 하기 위해서는
    # beta0를 빼고 넣는다. 이때, beta0에 y값의 평균이 들어감 
    return (a.transpose() @ a) + 500*np.abs(beta[1:]).sum()

# 들어가는 숫자는 beta0, beta1, beta2
# 최솟값 나오는 beta 값(람다 3)
line_perform_lasso([8.14, 0.96, 0])
# 예측식: y_hat = 8.14 + 0.96 * x1 + 0 * x2

# 최솟값 나오는 beta 값(람다 500)
line_perform_lasso([17.74, 0, 0])
# 예측식: y_hat = 17.74 + 0 * x1 + 0 * x2

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)
result.fun # 최소값
result.x # beta0, beta1, beta2

# 람다 값에 따라 변수 선택 됨
# x 변수가 추가 되면, train_x에서는 성능 항상 좋아짐
# x 변수가 추가 되면, valid_x에서는 좋아졌다가 나빠짐
# 나빠지는 순간이 overfitting 이라 함
# 따라서, overfitting을 방지하려면 
# x 변수 추가하는 걸 멈추는 시점을 알아야 함
# 람다 0부터 시작: 내가 가진 모든 변수를 넣음
# 점점 람다를 증가: 변수가 하나씩 빠짐
# 따라서, 람다를 점점 증가하다가 valid_x에서 가장 성능이 좋은 람다 선택
# 람다가 고정이 되면 그때 변수가 선택됨을 의미
# (X^T X)^-1 -> x의 칼럼이 선형 종속이면 다중공산성 존재

#=========================================
# Lasso lambda 정하기
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

np.random.seed(2024)
# -4에서 4까지의 균등 분포에서 30개의 무작위 값을 생성
x = uniform.rvs(size=30, loc=-4, scale=8)
# x에 대한 sin 함수를 계산하고, 정규 분포에서 랜덤 노이즈를 추가하여 y 값을 생성
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
# x와 y 값을 포함하는 데이터 프레임
df = pd.DataFrame({
    "y" : y,
    "x" : x
})

# 데이터 프레임의 처음 20개 행을 train으로 선택
train_df = df.loc[:19]

# x의 제곱부터 20제곱까지의 파생 변수를 추가
for i in range(2, 21):
    # x의 i 제곱을 계산하여 x2, x3, ..., x20 열을 추가
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

# 나머지 10개의 행을 valid로 선택
valid_df = df.loc[20:]

# 검증 데이터에도 같은 방식으로 파생 변수를 추가
for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_y = valid_df["y"]

from sklearn.linear_model import Lasso

# 각각 train와 valid에 대한 성능 평가 결과를 저장하기 위한 벡터 생성
val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

# 0.00부터 0.99까지의 alpha 값을 설정하여 Lasso 회귀 모델을 학습
for i in np.arange(0, 100):
    # Lasso 회귀 모델을 alpha 값으로 생성
    model= Lasso(alpha=i*0.01)
    # 모델을 train으로 학습
    model.fit(train_x, train_y)

    # 모델 성능(예측 값 계산)
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    # 예측 오류(제곱 오차 합)를 계산
    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    # train과 valid에 대한 성능 결과를 저장
    tr_result[i]=perf_train
    val_result[i]=perf_val

import seaborn as sns

# Lasso의 alpha 값에 대한 train, valid 
# 성능 결과를 포함하는 데이터 프레임
df = pd.DataFrame({
    'l': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='l', y='tr') # train
sns.scatterplot(data=df, x='l', y='val', color='red') # valid
plt.xlim(0, 0.4)

# 검증 데이터 성능의 최솟값
np.min(val_result)

# alpha를 0.03로 선택!
# 검증 데이터 성능이 가장 낮은 인덱스를 찾는 함수
np.argmin(val_result) # 3이 나오는데, 이는 4번째 행
# 0 포함이니까 인덱스 번호 3 -> 0 0.01 0.02 0.03(4번째 행)
# 최적의 alpha 값을 찾기 위해 인덱스를 사용하여 alpha 값을 출력
np.arange(0, 1, 0.01)[np.argmin(val_result)]

#=========================================
# minimize로 Ridge beta 구하기(패널티 있는 손실함수)
from scipy.optimize import minimize

# (y-X*beta_hat)T(y-X*beta_hat): 절대값 -> 함수화
# a: (y-X*beta_hat)
def line_perform_ridge(beta):
    beta = np.array(beta).reshape(3, 1)
    a = (y - matX @ beta)
    return (a.transpose() @ a) + 3*(beta**2).sum()

# 들어가는 숫자는 beta0, beta1, beta2
# 최솟값 나오는 beta 값
line_perform_ridge([0.86627049, 0.91084704, 0.61961358])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_ridge, initial_guess)
result.fun # 최소값
result.x # beta0, beta1, beta2