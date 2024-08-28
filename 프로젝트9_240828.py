# Lasso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드(1번)
berry_train = pd.read_csv('Blueberry/train.csv')
berry_test = pd.read_csv('Blueberry/test.csv')
sub_df = pd.read_csv('Blueberry/sample_submission.csv')

berry_train.isna().sum()
berry_test.isna().sum()

train_x = berry_train.drop("yield", axis = 1)
train_y = berry_train["yield"]

test_x = berry_test.drop("yield", axis=1, errors='ignore')

kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, 
                                     cv = kf,
                                     n_jobs = -1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0.006, 0.007, 0.00001)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 선형 회귀 모델 생성(8번)
model = Lasso(alpha=0.006769999999999969) # valid는 lambda 알기 위해서 쓰는 것

# 모델 학습
model.fit(train_x, train_y) # train 적용

pred_y=model.predict(test_x) # test로 predict 하기

# SalePrice 바꿔치기
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("submission/sample_submission_berry_1.csv", index=False)

#====================================================
# Ridge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드(1번)
berry_train = pd.read_csv('Blueberry/train.csv')
berry_test = pd.read_csv('Blueberry/test.csv')
sub_df = pd.read_csv('Blueberry/sample_submission.csv')

berry_train.isna().sum()
berry_test.isna().sum()

train_x = berry_train.drop("yield", axis = 1)
train_y = berry_train["yield"]

test_x = berry_test.drop("yield", axis=1, errors='ignore')

kf = KFold(n_splits=15, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, 
                                     cv = kf,
                                     n_jobs = -1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0.020, 0.025, 0.0001)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Ridge Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 선형 회귀 모델 생성(8번)
model = Ridge(alpha=0.024199999999999975) # valid는 lambda 알기 위해서 쓰는 것

# 모델 학습
model.fit(train_x, train_y) # train 적용

pred_y=model.predict(test_x) # test로 predict 하기

# SalePrice 바꿔치기
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("submission/sample_submission_berry_2.csv", index=False)

#==============================================
# 선형회귀
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 데이터 로드(1번)
berry_train = pd.read_csv('Blueberry/train.csv')
berry_test = pd.read_csv('Blueberry/test.csv')
sub_df = pd.read_csv('Blueberry/sample_submission.csv')

berry_train.isna().sum()
berry_test.isna().sum()

train_x = berry_train.drop("yield", axis = 1)
train_y = berry_train["yield"]

test_x = berry_test.drop("yield", axis=1, errors='ignore')

model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
pred_y=model.predict(test_x)

# yield 바꿔치기
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("submission/sample_submission_berry_3.csv", index=False)

#=======================================
ridge = pd.read_csv("Data/ridge.csv")
lasso = pd.read_csv("Data/lasso.csv")

yield_ri = ridge["yield"]
yield_la = lasso["yield"]

yield_total = ((yield_ri * 7) + (yield_la * 3))/10

sub_df["yield"] = yield_total

# csv 파일로 내보내기
sub_df.to_csv("submission/submission_total.csv", index=False)
