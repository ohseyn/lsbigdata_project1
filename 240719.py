# E[X]
sum(np.arange(4) * np.array([1, 2, 2, 1])/6)

# 전체 나온 것의 x가 나온 비율(p)
# https://onlinestatbook.com/stat_sim/sampling_dist/

import numpy as np
import matplotlib.pyplot as plt

# 예제 numpy 배열 생성
x = np.random.rand(10000, 5).mean(axis=1)
x = np.random.rand(50000).reshape(-1, 5).mean(axis=1)
plt.hist(x, bins=30, alpha=0.7, color="blue")
plt.grid(False)
plt.show()
plt.clf()

# 히스토그램(빈도표) 그리기
# bins(가로축 구간 개수, 폭), grid(경계)
plt.hist(data, bins=4, alpha=0.7, color="blue")
plt.title("Histogram of Numpy Vector")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()
plt.clf()
