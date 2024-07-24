# E[X]
sum(np.arange(4) * np.array([1, 2, 2, 1])/6)

# 전체 나온 것의 x가 나온 비율(p)
# https://onlinestatbook.com/stat_sim/sampling_dist/

import numpy as np
import matplotlib.pyplot as plt

# 1번
x = np.random.rand(10000, 5).mean(axis=1)
# 2번
x = np.random.rand(50000).reshape(-1, 5)
# rand 함수가 가능한 이유는 기본으로 0과 1 사이 난수를 뽑아내기 때문 
# 만약 0과 1 사이가 아닌 다른 값으로 할 거라면 * 를 이용하면 된다
x = np.random.rand(50000).reshape(-1, 5)*30
x_mean = x.mean(axis=1)

# 히스토그램(빈도표) 그리기
# bins(가로축 구간 개수, 폭), grid(경계)
plt.hist(data, bins=30, alpha=0.7, color="blue")
plt.title("Histogram of Numpy Vector")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()
plt.clf()

x = np.arange(33)
sum(x)/33 # 기댓값
sum((x-16) * 1/33) # E[X-E(X)]
(x-16)**2 # (X-E(X))**2

np.unique((x-16)**2)
np.unique((x-16)**2) * (2/33)
sum(np.unique((x-16)**2) * (2/33)) # E[(X-E(X))**2]

# E[X**2]
sum(x**2 * (1/33))

# E[(X-E(X))**2] = E[X**2 - 2*X*E(X) + E(X)**2]
# Var(X) = E[X**2] - (E[X])**2
sum(x**2 * (1/33)) - (16**2)

x = np.arange(4)
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
pro_x

# 기대값(E(X))
Ex = sum(x * pro_x)
# 제곱 기대값(E(X**2))
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum(((x - Ex)**2)*pro_x)

x = np.arange(99)
pro_x = np.concatenate((np.arange(1, 51), np.arange(49, 0, -1)))/2500
Ex = sum(x * pro_x)
sum(((x - Ex)**2)*pro_x)

y = np.arange(0, 8, 2) # np.arange(4)*2
pro_y = np.array([1/6, 2/6, 2/6, 1/6])
Ey = sum(y * pro_y)
sum(((y - Ey)**2)*pro_y)

# 표준편차(9.52가 표준편차, 표준편차**2=분산)
np.sqrt(9.52**2/25)
