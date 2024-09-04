# 의사결정트리
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

df=penguins.dropna()
df=df[["bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={'bill_length_mm': 'y',
                   'bill_depth_mm': 'x'})

# x=15 기준으로 나눴을 때, 데이터 포인트가 몇개씩 나눠지는지
n1=df.query("x < 15").shape[0] # sum(df["x"]<15) # 1번 그룹
n2=df.query("x >= 15").shape[0] # sum(df["x"]>=15) # 2번 그룹

# 1번 그룹, 2번 그룹 예측
y_hat1=df.query("x < 15").mean()[0]
y_hat2=df.query("x >= 15").mean()[0]

# 각 그룹 MSE
mse1=np.mean((df.query("x < 15")["y"]-y_hat1)**2)
mse2=np.mean((df.query("x >= 15")["y"]-y_hat2)**2)

# x=15의 MSE 가중평균
# (mse1+mse2)/2
((mse1*n1)+(mse2*n2))/(len(df))

#=================================================
# x=20의 MSE 가중평균
n1=df.query("x < 20").shape[0] 
n2=df.query("x >= 20").shape[0]
y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]
mse1=np.mean((df.query("x < 20")["y"]-y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"]-y_hat2)**2)
((mse1*n1)+(mse2*n2))/(len(df))

#================================================
# 원래 MSE
np.mean((df["y"]-df["y"].mean())**2)

#================================================
# 기준값 x를 넣으면 MSE값이 나오는 함수
def my_mse(x):
    n1=df.query(f"x<{x}").shape[0] 
    n2=df.query(f"x>={x}").shape[0]
    y_hat1=df.query(f"x<{x}").mean()[0]
    y_hat2=df.query(f"x>={x}").mean()[0]
    mse1=np.mean((df.query(f"x<{x}")["y"]-y_hat1)**2)
    mse2=np.mean((df.query(f"x>={x}")["y"]-y_hat2)**2)
    return ((mse1*n1)+(mse2*n2))/(len(df))
my_mse(20)

df["x"].min()
df["x"].max()

# 13~22 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용하여 가장 작은 MSE가 나오는 x 값 찾기
x_values=np.arange(13.2, 21.4, 0.01)
# 각 기준값 x에 대해 MSE를 계산하고 저장할 공간을 미리 할당
result=np.repeat(0.0, 820)

for i in range(820):
    result[i] = my_mse(x_values[i])

np.argmin(result)
x_values[np.argmin(result)]

# 두번째 나눌때 기준값을 얼마가 되어야 하는지
# 깊이 2 트리의 기준값 두개
group1 = df.query("x < 16.41")# 1번 그룹
group2 = df.query("x > 16.41")  # 2번 그룹

def my_mse(data, x):
    n1=data.query(f"x<{x}").shape[0] 
    n2=data.query(f"x>={x}").shape[0]
    y_hat1=data.query(f"x<{x}").mean()[0]
    y_hat2=data.query(f"x>={x}").mean()[0]
    mse1=np.mean((data.query(f"x<{x}")["y"]-y_hat1)**2)
    mse2=np.mean((data.query(f"x>={x}")["y"]-y_hat2)**2)
    return ((mse1*n1)+(mse2*n2))/(n1+n2)

x_values1 = np.arange(group1['x'].min()+0.01, group1['x'].max(), 0.01)
result1 = np.repeat(0.0, len(x_values1))
for i in range(0, len(x_values1)):
    result1[i] = my_mse(group1, x_values1[i])
x_values1[np.argmin(result1)] # 14.01

x_values2 = np.arange(group2['x'].min() + 0.01, group2['x'].max(), 0.01)
result2 = np.repeat(0.0, len(x_values2))
for i in range(0, len(x_values2)):
    result2[i] = my_mse(group2, x_values2[i])
x_values2[np.argmin(result2)] # 19.4

#==============================================
# x,y 산점도를 그리고 빨간 평행선 4개 그리기
df.plot(kind="scatter", x="x", y="y")
threshold=[14.01, 16.42, 19.4]
# df["x"] 값을 threshold에 따라 그룹으로 분할
# 각 값은 해당 기준값에 따라 그룹 번호가 매겨짐
df["group"]=np.digitize(df["x"], threshold)
# 각 그룹의 평균 y 값
y_mean=df.groupby("group").mean()["y"]

k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.41, 100)
k3=np.linspace(16.41, 19.4, 100)
k4=np.linspace(19.4, 22, 100)

import matplotlib.pyplot as plt
plt.axvline(x=16.41, color='green', linestyle=':')
plt.axvline(x=14.01, color='green', linestyle=':')
plt.axvline(x=19.4, color='green', linestyle=':')
# 그룹별 x 100개, 각 그룹의 평균 y값 100개
plt.plot(k1, np.repeat(y_mean[0], 100), color="red")
plt.plot(k2, np.repeat(y_mean[1], 100), color="red")
plt.plot(k3, np.repeat(y_mean[2], 100), color="red")
plt.plot(k4, np.repeat(y_mean[3], 100), color="red")

#================================================
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 랜덤 시드 고정 (재현성을 위해)
np.random.seed(42)

# x 값 생성: -10에서 10 사이의 100개 값
x = np.linspace(-10, 10, 100)

# y 값 생성: y = x^2 + 정규분포 노이즈
y = x ** 2 + np.random.normal(0, 10, size=x.shape)

# 데이터프레임 생성
df = pd.DataFrame({'x': x, 'y': y})

# 데이터 시각화
plt.scatter(df['x'], df['y'])

# 입력 변수와 출력 변수 분리
X = df[['x']]  # 독립 변수는 2차원 형태로 제공되어야 함
y = df['y']

# 학습 데이터와 테스트 데이터로 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# 디시전 트리 회귀 모델 생성 및 학습
# max_depth 개수: 2^n
model = DecisionTreeRegressor(random_state=42,
                              max_depth=6,
                              min_samples_split=10)
model.fit(X_train, y_train)

df_x=pd.DataFrame({"x": x})

# -10, 10까지 데이터에 대한 예측
y_pred = model.predict(X_test)

plt.scatter(X_test, y_pred, color="red")

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 테스트 데이터의 실제 값과 예측 값 시각화
plt.scatter(X_test, y_test, color='blue', label='Actual Values')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
