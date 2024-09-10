import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

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