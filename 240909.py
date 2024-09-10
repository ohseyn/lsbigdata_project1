import pandas as pd
import numpy as np
admission_data = pd.read_csv("Data/admission.csv")

# GPA: 학점
# GRE: 대학원 입학시험(영어, 수학)
p_hat = admission_data['admit'].mean()
Odds = p_hat / (1 - p_hat)

# P(A) > 0.5 -> 오즈비: 무한대에 가까워짐
# P(A) = 0.5 -> 오즈비 = 1
# P(A) < 0.5 -> 오즈비: 0에 가까워짐
# 확률의 오즈비가 갖는 값의 범위: 0~무한대
# P(A)와 오즈비는 비례 관계

unique_ranks = admission_data['rank'].unique()
# 랭크별 합격률
grouped_data = admission_data\
                .groupby('rank', as_index=False)\
                .agg(p_admit=('admit', 'mean'))
# 랭크별 합격률의 오즈비
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])

# 확률의 오즈비가 3 -> P(A)?
# P(A) = 3/4

import seaborn as sns
# sns.stripplot(data=admission_data,
#               x='rank', y='admit', jitter=0.3, alpha=0.3)

sns.scatterplot(data=grouped_data,
              x='rank', y='p_admit')

sns.regplot(data=grouped_data, x='rank', y='p_admit')

odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])

sns.regplot(data=odds_data, x='rank', y='log_odds')

import statsmodels.api as sm
# log_adds: 종속변수
# rank: 독립변수
model = sm.formula.ols("log_odds ~ rank", data=odds_data).fit()
print(model.summary())

np.exp(-0.5675)

# rank, gender 더미 코딩
admission_data = pd.read_csv("Data/admission.csv")
# admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')
model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data=admission_data).fit()
print(model.summary())

# 입학할 확률의 오즈가 np.exp(0.7753)
# GPA: 3.5 GRE: 500 GENDER: FEMALE RANK: 2
# 합격할 확률?
log_odds = (-3.4075 + (-0.0576* 0) + (0.0023 * 500) + (0.7753 * 3.5) + (-0.5614*2))
odds = np.exp(log_odds) # 0.51
p_hat = odds / (odds + 1) # 합격 확률(0.33)

# GPA가 1 증가하면 합격확률?
log_odds = (-3.4075 + (-0.0576* 0) + (0.0023 * 500) + (0.7753 * 4.5) + (-0.5614*2))
odds = np.exp(log_odds) # 1.11
p_hat = odds / (odds + 1) # 합격 확률(0.52)

# GPA: 3 GRE: 450 GENDER: FEMALE RANK: 2
log_odds = (-3.4075 + (-0.0576* 0) + (0.0023 * 450) + (0.7753 * 3.0) + (-0.5614*2))
odds = np.exp(log_odds) # 0.31
p_hat = odds / (odds + 1) # 합격 확률(0.23)

from scipy.stats import norm

2*(1-norm.cdf(2.123, loc=0, scale=0.1)) # 0.0
(0.25-0)/(2/np.sqrt(17)) # 0.5153882032022076
2*(1-norm.cdf(0.5153882032022076, loc=0, scale=1)) # 0.6062817742964275

stat_value=-2*(-249.99 - (-229.69)) # 40.60000000000002

from scipy.stats import chi2

1-chi2.cdf(stat_value, df=4) # df=변수갯수

#===============================================
# 1. 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성
leukemia_df = pd.read_csv('Data/leukemia_remission.txt', delimiter='\t')
model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP", data=leukemia_df).fit()
print(model.summary())

# 2. 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명
# LLR p-value가 0.05보다 작아서 모델 자체는 유의

# 3. 유의수준이 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수인지
# 2개. LI, TEMP

# 4. 다음 환자에 대한 오즈는 얼마? 
# CELL: 65% SMEAR: 45% INFIL: 55%
# LI: 1.2 BLAST: 1.1 세포/μL TEMP: 0.9
log_odds = (64.2581 + (30.8301* 0.65) + (24.6863 * 0.45) + (-24.9745 * 0.55) + (4.3605*1.2) + (-0.0115 * 1.1) + (-100.1734 * 0.9))
odds = np.exp(log_odds) # 0.03817459641135519
p_hat = odds / (odds + 1) # 0.03677088280074742

# 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마?
1-p_hat # 0.9632291171992526

# 6. TEMP 변수의 계수는 얼마이며, 
# 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명
# -100.1734
# np.exp(100.1734)은 오즈비로 TEMP가 1단위 증가할 때 마다 
# 백혈병 관측이 안됨에 대한 오즈가 오즈비만큼 증가하는 것
# 즉 온도가 올라갈수록 백혈병 세포가 관측 안 될 확률이 증가하는 것

# 7. CELL 변수의 99% 오즈비에 대한 신뢰구간
30.8301/52.135
2*(1-norm.cdf(0.5913512995108853, 0, 1))
30.8301 + 2.58*52.135 # 165.33839999999998
30.8301 - 2.58*52.135 # -103.67819999999999

# 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 
# 50% 이상인 경우 1로 처리하여, 혼동 행렬 구하기
