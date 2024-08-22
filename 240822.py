import numpy as np
import matplotlib.pyplot as plt

# y=a*x^2 + b*x+ c
a=2
b=3
c=5

x = np.linspace(-8, 8, 100)
y = a*x**2 + b*x + c
plt.plot(x, y, color="black")

# y=a*x**3 + b*x**2 + c*x + d
# y=a*x**4 + b*x**3 + c*x**2 + d*x + e
a=1
b=0
c=-10
d=0
e=10

x = np.linspace(-4, 4, 100)
# y = a*x**3 + b*x**2 + c*x + d
y = a*x**4 + b*x**3 + c*x**2 + d*x + e
plt.plot(x, y, color="black")

#====================================

# 데이터 만들기
from scipy.stats import norm
from scipy.stats import uniform

# 검정색 곡선
# k = np.linspace(-4, 4, 200)
# sin_y = np.sin(k)

# 파란색 점
# uniform.rvs(loc(구간 시작점), scale(구간 길이), size)
x = uniform.rvs(size=20, loc=-4, scale=8) # 랜덤으로 x 자리 뽑음
# 그 위치를 sin에 넣어서 y 좌표도 찍음
# norm.rvs는 노이즈(앱실론)
# 정규분포 norm.rvs(loc=0, scale=1, size=None, random_state=None)
y = np.sin(x) + norm.rvs(size=20, loc=0, scale=0.3)

# plt.plot(k, sin_y, color="black")
plt.scatter(x, y, color="blue")

#===========================================

np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "x":x, "y": y
})

train_df = df.loc[:19]
test_df = df.loc[20:]

from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 생성
model = LinearRegression()

x=train_df[["x"]]
y=train_df["y"]

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

reg_line = model.predict(x)

plt.plot(x, reg_line, color="red")
plt.scatter(train_df["x"], train_df["y"], color="blue")

#====================================

train_df["x2"]=train_df["x"]**2

# 2차 곡선 회귀
x=train_df[["x", "x2"]]
y=train_df["y"]

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2
})

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color="red")
plt.scatter(train_df["x"], train_df["y"], color="blue")

#============================================

train_df["x3"]=train_df["x"]**3

# 3차 곡선 회귀
x=train_df[["x", "x2", "x3"]]
y=train_df["y"]

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3
})

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color="red")
plt.scatter(train_df["x"], train_df["y"], color="blue")

#===============================================
# x를 쓰고 싶다면

train_df["x3"]=train_df["x"]**3

# 3차 곡선 회귀
x=train_df[["x", "x2", "x3"]]
y=train_df["y"]

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3
})

reg_line = model.predict(x.sort_values("x"))

plt.plot(x.sort_values("x")[["x"]], reg_line, color="red")
plt.scatter(train_df["x"], train_df["y"], color="blue")

#============================================

train_df["x4"]=train_df["x"]**4

# 4차 곡선 회귀
x=train_df[["x", "x2", "x3", "x4"]]
y=train_df["y"]

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3, "x4": k**4
})

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color="red")
plt.scatter(train_df["x"], train_df["y"], color="blue")

#============================================

train_df["x5"]=train_df["x"]**5
train_df["x6"]=train_df["x"]**6
train_df["x7"]=train_df["x"]**7
train_df["x8"]=train_df["x"]**8
train_df["x9"]=train_df["x"]**9

# 9차 곡선 회귀
x=train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]
y=train_df["y"]

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3, "x4": k**4,
    "x5": k**5, "x6": k**6, "x7": k**7,
    "x8": k**8, "x9": k**9
})

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color="red")
plt.scatter(train_df["x"], train_df["y"], color="blue")

#===========================================

train_df["x10"]=train_df["x"]**10
train_df["x11"]=train_df["x"]**11
train_df["x12"]=train_df["x"]**12
train_df["x13"]=train_df["x"]**13
train_df["x14"]=train_df["x"]**14
train_df["x15"]=train_df["x"]**15

# 15차 곡선 회귀
x=train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", 
            "x8", "x9", "x10", "x11", "x12", "x13", 
            "x14", "x15"]]
y=train_df["y"]

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3, "x4": k**4,
    "x5": k**5, "x6": k**6, "x7": k**7,
    "x8": k**8, "x9": k**9, "x10": k**10,
    "x11": k**11, "x12": k**12, "x13": k**13,
    "x14": k**14, "x15": k**15
})

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color="red")
plt.scatter(train_df["x"], train_df["y"], color="blue")

#======================================
# test에 있는 걸로 y값 추측해보기!!

# 9차 곡선 회귀
x=train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]
y=train_df["y"]

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3, "x4": k**4,
    "x5": k**5, "x6": k**6, "x7": k**7,
    "x8": k**8, "x9": k**9
})

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color="red")
plt.scatter(train_df["x"], train_df["y"], color="blue")

# test에 있는 데이터로 성능 테스트 하기
test_df["x2"]=test_df["x"]**2
test_df["x3"]=test_df["x"]**3
test_df["x4"]=test_df["x"]**4
test_df["x5"]=test_df["x"]**5
test_df["x6"]=test_df["x"]**6
test_df["x7"]=test_df["x"]**7
test_df["x8"]=test_df["x"]**8
test_df["x9"]=test_df["x"]**9

x=test_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]

y_hat = model.predict(x)

sum((test_df["y"] - y_hat)**2)

#==========================================
# 20차 모델 성능

# 각 반복에서, 데이터프레임 train_df에 새로운 열을 추가
# 새로 추가되는 열의 이름은 "x{i}" 형식
# {i}는 현재 반복 중의 숫자
# train_df["x"]의 값들을 i제곱한 값을 새로운 열에 저장
for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i

j=3
f"x{j:.2f}"

# train_df 데이터프레임에서 x 열과 앞서 추가한 "x2"부터 "x20"까지의 열들을 선택
# 선택된 열들로 새로운 데이터프레임 x를 생성
x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
y=train_df["y"]

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

k = np.linspace(-4, 4, 200)
# k의 값을 제곱부터 20제곱까지 계산하여 데이터프레임을 생성
# 딕셔너리 컴프리헨션을 사용하여 열 이름과 그 값을 지정
# f"x{i}"는 열의 이름을 문자열 형식
# k**i는 k의 i제곱을 계산
df_k = pd.DataFrame({
    "x": k,  
    # **는 딕셔너리의 키-값 쌍을 개별적인 인수로 풀어내어 
    # 함수 호출 시에 각 항목을 별도로 전달
    **{f"x{i}": k**i for i in range(2, 21)} 
})

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color="red")
plt.scatter(train_df["x"], train_df["y"], color="blue")

# 20차 모델 성능
for i in range(2, 21):
    test_df[f"x{i}"] = test_df["x"] ** i

x = test_df[["x"] + [f"x{i}" for i in range(2, 21)]]

y_hat = model.predict(x)

# 실제에서는 test_df["y"]를 모름
# 그럼 이걸 확인하려면 train을 쪼개야 함
sum((test_df["y"] - y_hat)**2)

#==========================================
## 필요한 데이터 불러오기
house_train=pd.read_csv("train.csv")
house_test=pd.read_csv("test.csv")
sub_df=pd.read_csv("sample_submission.csv")

house_train.shape
house_test.shape 
train_n=house_train.shape[0]

# 합친 이유는 한 번에 dummies 하려고
df=pd.concat([house_train, house_test], 
             axis=0, ignore_index=True)

# 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   판다스 시리즈
# house_train[["GrLivArea"]] 판다스 프레임

neighborhood_dummies = pd.get_dummies(
    df["Neighborhood"],
    drop_first=True
    )

# pd.concat([df_a, df_b], axis=1)
x= pd.concat([df[["GrLivArea", "GarageArea"]], 
             neighborhood_dummies], axis=1)
y = df["SalePrice"]

train_x=x.iloc[:train_n,]
test_x=x.iloc[train_n:,]
train_y=y[:train_n]

# Validation set(모의고사) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), 
                             size=438, replace=False)
valid_x=train_x.loc[val_index] # 30%
valid_y=train_y.loc[val_index] # 30%
train_x=train_x.drop(val_index) # 70%
train_y=train_y.drop(val_index) # 70%

# 이상치 제거
train_x=train_x.query("GrLivArea <= 4500")

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 성능 측정
y_hat=model.predict(valid_x) # valid_y에 대한 추정값
np.sqrt(np.mean((valid_y-y_hat)**2))