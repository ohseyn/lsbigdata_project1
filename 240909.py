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
