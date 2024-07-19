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
