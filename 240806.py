import numpy as np
import pandas as pd

tab3 = pd.read_csv("Data/tab3.csv")
tab1 = pd.DataFrame({"id": np.arange(1, 13), "score": tab3["score"]})
tab2 = tab1.assign(gender=["female"]*7+["male"]*5)

# 1 표본 t 검정(그룹 1개)
# 귀무가설 vs 대립가설
# H0: mu = 10 vs Ha: mu != 10
# 유의수준 5%로 설정
from scipy.stats import ttest_1samp

# 10과 같지 않다라고 대립가설이 정해져서 two-sided로 해야함
# p_value는 양쪽을 더한 값
result = t_statistic, p_value = ttest_1samp(tab1["score"], popmean=10, alternative='two-sided')
t_value = result[0] # t_statistic(검정통계량) = result.pvalue
p_value = result[1] # p_value(유의확률) = result.statistic
tab1["score"].mean() # 표본평균
result.df # 자유도
ci = result.confidence_interval(confidence_level=0.95) # 신뢰구간
ci[0] # low
ci[1] # high

# 귀무가설이 참(mu = 10)일 때, 표본평균(11.53)이 관찰될 확률이 6.48%(p_value, 0.0648)이므로 
# 우리가 생각하는 보기 힘들다고 판단하는 기준인 0.05(유의수준)보다 크므로
# 귀무가설을 거짓이라 판단하기 어렵다.
# 따라서, 유의확률 > 유의수준 이므로 귀무가설 기각 못 한다.

# 2 표본 t 검정 (그룹 2개) - 분산 같을 때 / 다를때
# 분산 같은 경우: 독립 2 표본 t 검정
# 분산 다를 경우: 웰치스 t 검정
# 귀무가설 vs 대립가설
# H0: mu_m = mu_f vs Ha: mu_m > mu_f
# 유의수준 1%로 설정, 두 그룹의 분산은 같다고 가정
from scipy.stats import ttest_ind

f_tab2 = tab2[tab2["gender"]=="female"]
m_tab2 = tab2[tab2["gender"]=="male"]
result = t_statistic, p_value = ttest_ind(f_tab2['score'], m_tab2['score'],
                                alternative="less", equal_var=True)

# equal_var=True 모표본의 분산이 그룹별로 동일함을 의미
# alternative="less" 이 부분은 대립 가설 보고 결정
# 의미는 대립가설이 첫번째 입력그룹의 평균이 두번째 입력 그룹 평균보다 작다.
# t_statistic, p_value = ttest_ind(m_tab2['score'], f_tab2['score'],
#                                alternative="greater", equal_var=True)
 
result.statistic
result.pvalue
result.df
ci = result.confidence_interval(confidence_level=0.95)

# 대응 표본 t 검정(짝지을 수 있는 표본)
# 귀무가설 vs 대립가설
# H0: mu_before = mu_after vs Ha: mu_after > mu_before
# H0: mu_d = 0 vs Ha: mu_d > 0
# mu_d = mu_after - mu_before
# 유의수준 1%로 설정

# mu_d에 대응하는 표본으로 변환
# 피벗 테이블 쓴 이유: after, before 이용하려고
tab3_data = tab3.pivot_table(index='id', 
                            columns='group',
                            values='score').reset_index()
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]

# 모평균이 0인 이유는 mu_d = 0 이렇게 가설을 설정했기 때문에
result = ttest_1samp(test3_data["score_diff"], popmean=0, alternative='greater')
t_value=result[0] # t 검정통계량
p_value=result[1] # 유의확률 (p-value)
