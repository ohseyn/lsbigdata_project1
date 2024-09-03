import numpy as np
import matplotlib.pyplot as plt

# 독립성 검정
mat_a = np.array([14, 4, 0, 10]).reshape(2,2)

# 귀무가설: 두 변수(흡연, 운동선수) 독립
# 대립가설: 두 변수 독립 X
from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a, 
                                         correction=False)
# np.sum((mat_a - expected)**2/expected)
chi2.round(3) # 검정 통계량
p.round(4) # p-value

# 유의수준 0.05라면, p 값이 0.05보다 작으므로, 귀무가설 기각
# 즉, 두 변수는 독립이 아니다
# X~chi2(1) 일 때, P(X > 15.556)
from scipy.stats import chi2

1-chi2.cdf(15.556, df=1)
p

#=========================================
# 동질성 검증
mat_b=np.array([[50, 30, 20], [45, 35, 20]])

chi2, p, df, expected = chi2_contingency(mat_b, 
                                         correction=False)
chi2.round(3) # 검정 통계량
p.round(4) # p-value
expected

#==========================================
# p.112 연습문제
# 귀무가설: 정당 지지와 핸드폰 사용 유무는 독립이다.
# 대립가설: 정당 지지와 핸드폰 사용 유무는 독립이 아니다.
mat_phone = np.array([49, 47, 15, 27, 32, 30]).reshape(3,2)

chi2, p, df, expected = chi2_contingency(mat_phone, 
                                         correction=False)
chi2.round(3) # 검정 통계량
p.round(4) # p-value
# 유의수준 0.05보다 p-value가 크므로, 귀무가설을 기각할 수 없다.
expected

#==========================================
# 적합도 검정
from scipy.stats import chisquare

# 자유도는 n-1로 바뀐다(7일이니 n-1 하면 6됨)
observed = np.array([13, 23, 24, 20, 27, 18, 15])
expected = np.repeat(20, 7)
statistic, p_value = chisquare(observed, f_exp=expected)
# p-value 0.2688이 유의수준 0.05 보다 크므로 귀무가설을 기각 못함.
# 즉, 요일 별 신생아 출생비율이 같다고 판단.

from scipy.stats import chi2

1-chi2.cdf(7.6000000000000005, df=6)

#=========================================
# 지역별 후보 지지율
# 귀무가설 : 선거구별 후보A의 지지율이 동일하다.
# 대립가설 : 선거구별 후보A의 지지율이 동일하지 않다.
mat_p = np.array([[176, 124], 
                  [193, 107], 
                  [159, 141]])

from scipy.stats import chi2

chi2, p, df, expected = chi2_contingency(mat_p, 
                                         correction=False)
chi2.round(3) # 검정 통계량
p.round(4) # p-value
# 유의수준 0.05라면, p 값이 0.05보다 작으므로, 귀무가설 기각
expected