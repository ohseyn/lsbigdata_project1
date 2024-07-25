from scipy.stats import bernoulli

# 베르누이 함수: 0과 1을 갖는 함수
# 확률질량함수 pmf: 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k, p)
# P(X=1)
bernoulli.pmf(1, 0.3)
# P(X=0)
bernoulli.pmf(0, 0.3)

# P(X = k | n, p)
# binom.pmf(k, n, p)
# n: 베르누이 확률변수 더한 개수
# p: 1이 나올 확률
# pmf: probability mass function 확률질량함수(막대기에 질량이 있다고 생각하기)
from scipy.stats import binom
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

# X ~ B(n, p)
result = [binom.pmf(x, 30, 0.3) for x in range(31)]
binom.pmf(np.arange(31), 30, 0.3)

import math
math.factorial(54)/(math.factorial(26)*math.factorial(28))
math.comb(54,26)

# nCr p**r * (1-p)**(n-r) == math.comb(n, r) * p**r * (1-p)**(n-r)
math.comb(2, 0) * 0.3**0 * (1-0.3)**2 # binom.pmf(0, 2, 0.3)
math.comb(2, 1) * 0.3**1 * (1-0.3)**1 # binom.pmf(1, 2, 0.3)
math.comb(2, 2) * 0.3**2 * (1-0.3)**0 # binom.pmf(2, 2, 0.3)

# X ~ B(n=10, p=0.36), P(X=4)
sum(binom.pmf(np.arange(0,5), 10, 0.36))
sum(binom.pmf(np.arange(3,9), 10, 0.36))
# X ~ B(30, 0.2), P(X<4 or X>=25)
sum(binom.pmf(np.concatenate((np.arange(0,4), np.arange(25, 31))), 30, 0.2)) # 4가 포함이 안 됨
sum(binom.pmf(np.arange(0,4), 30, 0.2)) + sum(binom.pmf(np.arange(25, 31), 30, 0.2))
sum(binom.pmf(np.arange(0, 31), 30, 0.2)) # 1.0
1 - sum(binom.pmf(np.arange(4,25), 30, 0.2)) # 1-P(4<=X<25)

# rvs 함수 (random variates sample)
# 표본 추출 함수
# X1 ~ Bernulli(p=0.3)
bernoulli.rvs(0.3)
# X2 ~ Bernulli(p=0.3)
bernoulli.rvs(0.3)
# X ~ B(2, 0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(2, 0.3, size = 1)
# X ~ B(30, 0.26), 표본 30개
binom.rvs(30, 0.26, size = 30)
binom.rvs(30, 0.26, size = 30).mean()

import seaborn as sns
prob_x = binom.pmf(np.arange(0,31), 30, 0.26)
sns.barplot(prob_x)
plt.show()
plt.clf()

import pandas as pd
x = np.arange(0, 31)
df = pd.DataFrame({"x": x, "prob": prob_x})
sns.barplot(data = df, x = "x", y = "prob")
plt.show()
plt.clf()

#=============================================================
#np.cumprod(np.arange(1,55))
# ln = loge(자연로그)
# log(a*b) = log(a)+ log(b)
math.log(math.factorial(54))
sum(np.log(np.arange(1,55)))
logf_54 = sum(np.log(np.arange(1,55)))
logf_28 = sum(np.log(np.arange(1,29)))
logf_26 = sum(np.log(np.arange(1,27)))
np.exp(logf_54 - (logf_28 + logf_26))
#=============================================================

import pandas as pd
house_df = pd.read_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/train.csv")
price_mean = house_df["SalePrice"].mean()
sub_df = pd.read_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission.csv")
sub_df["SalePrice"] = price_mean
sub_df.to_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission.csv", 
                index = False) # index = False 안 하면 열 하나가 추가 됨 
