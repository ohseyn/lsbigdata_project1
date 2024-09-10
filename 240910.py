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
# 모든 베타가 0이라고 할 수 없기에 유의하다.

# 3. 유의수준이 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수인지
# 2개. LI, TEMP

# 4. 다음 환자에 대한 오즈는 얼마? 
# CELL: 65% SMEAR: 45% INFIL: 55%
# LI: 1.2 BLAST: 1.1 세포/μL TEMP: 0.9
log_odds = (64.2581 + (30.8301* 0.65) + (24.6863 * 0.45) + (-24.9745 * 0.55) + (4.3605*1.2) + (-0.0115 * 1.1) + (-100.1734 * 0.9))
odds = np.exp(log_odds) # 0.03817459641135519

# 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마?
p_hat = odds / (odds + 1) # 0.03677088280074742

# 6. TEMP 변수의 계수는 얼마이며, 
# 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명

# -100.1734
# np.exp(-100.1734): 0에 가까운 값 
# 이는 체온이 1단위 상승할 때 백혈병 세포가 관측되지 않을 확률이 
# 거의 없어지는 것을 의미(오즈비만큼 변동)
# 온도가 높아질수록 백혈병 세포가 관측될 확률 높아짐.

# 7. CELL 변수의 99% 오즈비에 대한 신뢰구간
30.8301/52.135
np.exp(30.8301 + 2.58*52.135) # 165.33839999999998
np.exp(30.8301 - 2.58*52.135) # -103.67819999999999

# 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 
# 50% 이상인 경우 1로 처리하여, 혼동 행렬 구하기
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

train = leukemia_df.drop(columns=('REMISS'))

y_pred = model.predict(train)
result = pd.DataFrame({'y_pred' : y_pred})
result['result'] = np.where(result['y_pred']>=0.5, 1,0)

conf_mat = confusion_matrix(y_true = leukemia_df['REMISS'], y_pred = result['result'], labels=[1,0])
p = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = ('1', '0'))
p.plot(cmap="Blues")

# 9. 해당 모델의 Accuracy는 얼마?
(5+15)/(5+3+4+15)

from sklearn.metrics import accuracy_score, f1_score
accuracy_score(leukemia_df['REMISS'], result['result'])

# 10. 해당 모델의 F1 Score를 구하기
precision = 5/(5+3)
recall = 5/(5+4)
f1 = 2 / (1/precision + 1/recall)

f1_score(leukemia_df['REMISS'], result['result'])

#=====================================================
# 앙상블, RandomForests
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging_model = BaggingClassifier(DecisionTreeClassifier(),
                                  n_estimators=100,
                                  max_samples=100,
                                  n_jobs=-1,
                                  random_state=42)

# n_estimator: Bagging에 사용될 모델 개수
# max_sample: 데이터셋 만들 때 뽑을 표본크기
# bagging_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=50,
                                max_leaf_node=16,
                                n_jobs=-1, random_state=42)