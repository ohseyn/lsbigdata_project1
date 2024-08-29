# 미분
import numpy as np
import matplotlib.pyplot as plt

a = 2
b = 1

x = np.linspace(-4, 8, 100)
# y=(x-2)**2+1
y = (x-a)**2 + 1

plt.plot(x, y, color="black")
plt.xlim(-4, 8)
plt.ylim(0, 15)

#===================================
# y=4x-11
line_y=4*x - 11
plt.plot(x, line_y, color="blue")

#====================================
# k 값 조정해서 해당하는 k 값에 대한 접선의 방정식 그릴 수 있음
k=2
l_slope = 2*k-4
f_k=(k-2)**2+1
l_intercept = f_k - l_slope * k

# y = slope*x + intercept
line_y = l_slope * x + l_intercept
plt.plot(x, line_y, color="red")

#=====================================
# y=x**2 경사하강법
# 초기값: 10, delta(Step): 0.9
x = 10
lstep= np.arange(100, 0, -1)*0.01
for i in range(100):
     x -= lstep[i]*(2*x)

#======================================
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의(-1에서 7까지)
x = np.linspace(-1, 7, 400) # 400개
y = np.linspace(-1, 7, 400) # 400개
# 160000개 점
# 배열을 격자 형태로 변환하여 2차원 그리드 생성
# 좌표의 집합
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산
z = (x - 3)**2 + (y - 4)**2 + 3

# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 표면 그래프를 그림
ax.plot_surface(x, y, z, cmap='viridis')

# ======================================
# 등고선 그래프
# 그래프의 등고선은 2차원 함수의 높이 값
x = np.linspace(-10, 10, 400) # 400개
y = np.linspace(-10, 10, 400) # 400개
# 160000개 점
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산
z = (x - 3)**2 + (y - 4)**2 + 3

# 등고선 그래프
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가

x=9
y=2
lstep=0.1
# plt.scatter(9, 2, color="red", s=10)
# x, y = np.array([x, y]) - lstep * np.array([2*x-6, 2*y-8])
# plt.scatter(x, y, color="red", s=10)

for i in range(100):
     x, y = np.array([x, y]) - lstep * np.array([2*x-6, 2*y-8])
     plt.scatter(x, y, color="red", s=10)
x,y

#========================================
# 등고선 그래프
# 그래프의 등고선은 2차원 함수의 높이 값
x = np.linspace(-10, 10, 400) # 400개
y = np.linspace(-10, 10, 400) # 400개
# 160000개 점
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산
z = (1-(x+y))**2 + (4-(x+2*y))**2 + (1.5-(x+3*y))**2 + (5-(x+4*y))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=40)  # levels는 등고선의 개수를 조절
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가

x=10
y=10
lstep=0.01

for i in range(100):
     x, y = np.array([x, y]) - lstep * np.array([8*x+20*y-23, 60*y+20*x-67])
     plt.scatter(x, y, color="red", s=10)
x,y
#=========================================
# 회귀직선 beta 찾기
# 그래프의 등고선은 2차원 함수의 높이 값
beta0 = np.linspace(-20, 20, 400) # 400개
beta1 = np.linspace(-20, 20, 400) # 400개
# 160000개 점
beta0, beta1 = np.meshgrid(beta0, beta1)

# 함수 f(x, y)를 계산
z = (1-(beta0+beta1))**2 + (4-(beta0+2*beta1))**2 + (1.5-(beta0+3*beta1))**2 + (5-(beta0+4*beta1))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta0, beta1, z, levels=80)  # levels는 등고선의 개수를 조절
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가

beta0=10
beta1=10
lstep=0.01

# for i in range(100):
#      beta0, beta1 = np.array([beta0, beta1]) - lstep * np.array([8*beta0+20*beta1-23, 60*beta1+20*beta0-67])
#      plt.scatter(beta0, beta1, color="red", s=10)
# beta0,beta1

for i in range(1000):
     beta0, beta1 = np.array([beta0, beta1]) - lstep * np.array([8*beta0+20*beta1-23, 60*beta1+20*beta0-67])
     plt.scatter(beta0, beta1, color="red", s=10)
beta0,beta1

import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.DataFrame({
     "x": np.array([1, 2, 3, 4]),
     "y": np.array([1, 4, 1.5, 5])
})
model = LinearRegression()
model.fit(df[["x"]], df["y"])

model.coef_
model.intercept_
