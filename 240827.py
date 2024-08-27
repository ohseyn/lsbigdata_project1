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

#===========================================
# 0.03으로 그려보기!
model= Lasso(alpha=0.03)
model.fit(train_x, train_y)

k=np.linspace(-4, 4, 801)

k_df = pd.DataFrame({
    "x" : k
})

for i in range(2, 21):
    k_df[f"x{i}"] = k_df["x"] ** i

reg_line = model.predict(k_df)

plt.plot(k_df["x"], reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"], color="blue")

#=========================================
# train 셋을 3개로 쪼개어 valid_set과 train_set 3개 만들기
# 각 세트에 대한 성능을 각 lambda 값에 대응하여 구하기
# 성능 평가 지표 3개 평균내어 그래프 다시 그리기

np.random.seed(2024)
# 시작 지점 −4, 길이 8인 균등분포에서 30개 샘플 뽑기
x = uniform.rvs(size=30, loc=-4, scale=8)
# x의 sin 값에 평균 0, 표준편차 0.3인 정규 분포의 난수를 더한 값
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y" : y, # 종속 변수
    "x" : x # 독립 변수
})

# x의 차수들 추가
for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

# 데이터프레임 무작위로 섞음
# 3개의 fold로 나누어 train_set와 valid_set 생성
def make_tr_val(fold_num, df):
    np.random.seed(2024)
    # np.random.choice(범위, 개수, 중복 허용 여부)
    myindex=np.random.choice(30, 30, replace=False)

    # fold_num에 따라 valid_set의 index를 결정
    val_index=myindex[(10*fold_num):(10*fold_num+10)]

    # valid set, train set
    valid_set=df.loc[val_index] 
    train_set=df.drop(val_index)

    train_X=train_set.iloc[:,1:] 
    train_y=train_set.iloc[:,0]

    valid_X=valid_set.iloc[:,1:]
    valid_y=valid_set.iloc[:,0]

    return (train_X, train_y, valid_X, valid_y)

from sklearn.linear_model import Lasso

# 3개의 fold와 각 fold당 1000개의 a값을 위한 검증 성능 결과를 저장할 배열
val_result_total=np.repeat(0.0, 3000).reshape(3, -1)
tr_result_total=np.repeat(0.0, 3000).reshape(3, -1)

for j in np.arange(0, 3):
    # 각 fold에 대해 train, valid 생성
    train_X, train_y, valid_X, valid_y = make_tr_val(fold_num=j, df=df)

    # 각 a값에 대해 train, valid 저장할 벡터 초기화
    val_result=np.repeat(0.0, 1000)
    tr_result=np.repeat(0.0, 1000)

    # 1000개의 a값을 반복하여 모델 학습
    for i in np.arange(0, 1000):
        # α 값을 0.01 단위로 조정하며 Lasso 모델 생성
        model= Lasso(alpha=i*0.01)
        # train 모델 학습
        model.fit(train_X, train_y)

        # train, valid 예측값 생성
        y_hat_train = model.predict(train_X)
        y_hat_val = model.predict(valid_X)

        # train, valid 성능 측정
        perf_train=sum((train_y - y_hat_train)**2)
        perf_val=sum((valid_y - y_hat_val)**2)

        # train, valid 성능 결과를 저장
        tr_result[i]=perf_train
        val_result[i]=perf_val

    # 각 fold에 대한 train, valid 성능 결과를 저장
    tr_result_total[j,:]=tr_result
    val_result_total[j,:]=val_result

import seaborn as sns

df = pd.DataFrame({
    # α 값을 0에서 10까지 0.01 단위로 생성하여 lambda 컬럼 생성
    'lambda': np.arange(0, 10, 0.01), 
    'tr': tr_result_total.mean(axis=0), # 열
    'val': val_result_total.mean(axis=0)
})

# seaborn을 사용하여 산점도 그리기
# sns.scatterplot(data=df, x='lambda', y='tr')
# 검증 성능(val)을 lambda 값에 따라 산점도로 시각화
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 10)

# alpha를 2.67로 선택!
np.argmin(val_result_total.mean(axis=0))
np.arange(0, 10, 0.01)[np.argmin(val_result_total.mean(axis=0))]
