import numpy as np

x = np.arange(2, 13)
prob = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36

# 1번
Ex = np.sum(x*prob)
Var = np.sum((x-Ex)**2 * prob)

# 2번
2*Ex+3
np.sqrt(4*Var)

#======================================
from scipy.stats import binom

# Y ~ B(3, 0.7)
# Y가 갖는 값에 대응하는 확률은?
# binom.pmf(y값, n(시행 횟수), p(확률))
binom.pmf(0, 3, 0.7)
binom.pmf(np.arange(0, 4, 1), 3, 0.7)
# binom.pmf(np.array([0, 1, 2, 3]), 3, 0.7)

# Y ~ B(20, 0.45)
# P(6 < Y <= 14)
sum(binom.pmf(np.arange(7, 15), 20, 0.45))
binom.cdf(14, 20, 0.45) - binom.cdf(6, 20, 0.45)

#==========================================
# X ~ N(30, 4^2)
# P(X > 24)
from scipy.stats import norm
1 - norm.cdf(24, loc=30, scale=4)

# 표본 8개 뽑아서 표본평균 계산
# P(28 < x < 29.7)
# X_bar ~ N(30, 4^2/8)
a = norm.cdf(29.7, loc=30, scale=np.sqrt(4**2/8))
b = norm.cdf(28, loc=30, scale=np.sqrt(4**2/8))
a-b

#==========================================
# 표준화 이용
mean = 30
s_var = 4/np.sqrt(8) # 자유도 

# 표준화
right_x = (29.7 - mean) / s_var 
left_x = (28 - mean) / s_var

# 표준정규분포
a = norm.cdf(right_x, 0, 1)
b = norm.cdf(left_x, 0, 1)
a-b

#====================================
# 자유도 7인 카이제곱분포 확률밀도 함수
from scipy.stats import chi2
import matplotlib.pyplot as plt

k = np.linspace(-2, 40, 500)
y = chi2.pdf(k, df=7)
plt.plot(k, y)

#======================================
# 독립성 검정
mat_a = np.array([14, 4, 0, 10]).reshape(2,2)

# 귀무가설: 두 변수(흡연, 운동선수) 독립
# 대립가설: 두 변수 독립 X
from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a)
chi2.round(3) # 검정 통계량
p.round(4) # p-value

# 유의수준 0.05라면, p 값이 0.05보다 작으므로, 귀무가설 기각
# 즉, 두 변수는 독립이 아니다
# X~chi2(1) 일 때, P(X > 12.6)
from scipy.stats import chi2

1-chi2.cdf(12.6, df=1)
p