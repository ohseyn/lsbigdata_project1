# E[X]
sum(np.arange(4) * np.array([1, 2, 2, 1])/6)

# 전체 나온 것의 x가 나온 비율(p)
# https://onlinestatbook.com/stat_sim/sampling_dist/

import numpy as np
import matplotlib.pyplot as plt

# 예제 numpy 배열 생성
dat = np.random.randn(1000)

# 히스토그램 그리기

plt.hist(data, bins=30, alpha=0.7, color="blue")
plt.title("Histogram of Numpy Vector")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
plt.clf()
