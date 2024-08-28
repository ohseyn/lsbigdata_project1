# 범주형, 숫자형 모두 다 입력값으로!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드(1번)
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
sub_df = pd.read_csv('sample_submission.csv')

# Nan 채우기
# 각 숫자 변수는 평균으로 채우기
# 각 범주형 변수는 최빈값으로 채우기
house_train.isna().sum()
house_test.isna().sum()

# 수치형만
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

# inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

# 범주형만
Categorical = house_train.select_dtypes(include = [object])
Categorical.isna().sum()
Cate_selected = Categorical.columns[Categorical.isna().sum() > 0]

for col in Cate_selected:
    # inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
    house_train[col].fillna(house_train[col].mode()[0], inplace=True)
house_train[Cate_selected].isna().sum()

#==========================================

quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

# inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
for col in quant_selected:
    house_test[col].fillna(house_test[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

# 범주형만
Categorical = house_test.select_dtypes(include = [object])
Categorical.isna().sum()
Cate_selected = Categorical.columns[Categorical.isna().sum() > 0]

for col in Cate_selected:
    # inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
    house_test[col].fillna(house_test[col].mode()[0], inplace=True)
house_test[Cate_selected].isna().sum()

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩(2,3번)
df = pd.concat([house_train, house_test], ignore_index=True)
# 변수들 중에서 good 이나 very good처럼 
# 순서가 있는 아이들은 숫자로 바꿔줘야하고, 
# 숫자로 되어있음에도 불구하고 범주형인 데이터도 있을 것이다. 
# 이런 친구들도 더미코딩을 해 줘야한다. 
# 이런 경우 우리들이 변수를 보고 수정을 해야하지만, 
# 시간이 없으니까 object 타입 열만 가져와서 해보자.
df = pd.get_dummies(
    df,
    # object 형태인 변수 다 가져옴
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )

# train / test 데이터셋으로 나누기(4번)
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

train_x = train_df.drop("SalePrice", axis=1)
train_y = train_df["SalePrice"]

# 인코딩 또는 디코딩 과정에서 발생할 수 있는 오류를 무시하고 
# 계속 진행할 수 있게 해줌
test_x = test_df.drop("SalePrice", axis=1, errors='ignore')

#=========================================
# Validation(교차검증) 만들기(10번)
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

# valid_result 한 칸 계산하는 함수
def rmse(model):
    # -- -> + 로 만들어준 것
    score = np.sqrt(-cross_val_score(model, 
                                     train_x, 
                                     train_y, 
                                     cv = kf,
                                     n_jobs=-1, 
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

alpha_values = np.arange(147, 149, 0.01)
# 막대기 만들어줌(valid_result)
mean_scores = np.zeros(len(alpha_values))

k=0
# for문으로 칸 다 채워서 막대기 채움
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

#==========================================
# 선형 회귀 모델 생성(8번)
model = Lasso(alpha=148.04999999999905) # valid는 lambda 알기 위해서 쓰는 것

# 모델 학습
model.fit(train_x, train_y) # train 적용

pred_y=model.predict(test_x) # test로 predict 하기

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("submission/sample_submission18.csv", index=False)

#========================================
# Lasso를 Ridge로 바꾸기!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드(1번)
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
sub_df = pd.read_csv('sample_submission.csv')

# Nan 채우기
# 각 숫자 변수는 평균으로 채우기
# 각 범주형 변수는 최빈값으로 채우기
house_train.isna().sum()
house_test.isna().sum()

# 수치형만
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

# inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

# 범주형만
Categorical = house_train.select_dtypes(include = [object])
Categorical.isna().sum()
Cate_selected = Categorical.columns[Categorical.isna().sum() > 0]

for col in Cate_selected:
    # inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
    house_train[col].fillna(house_train[col].mode()[0], inplace=True)
house_train[Cate_selected].isna().sum()

#==========================================

quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

# inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
for col in quant_selected:
    house_test[col].fillna(house_test[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

# 범주형만
Categorical = house_test.select_dtypes(include = [object])
Categorical.isna().sum()
Cate_selected = Categorical.columns[Categorical.isna().sum() > 0]

for col in Cate_selected:
    # inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
    house_test[col].fillna(house_test[col].mode()[0], inplace=True)
house_test[Cate_selected].isna().sum()

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩(2,3번)
df = pd.concat([house_train, house_test], ignore_index=True)
# 변수들 중에서 good 이나 very good처럼 
# 순서가 있는 아이들은 숫자로 바꿔줘야하고, 
# 숫자로 되어있음에도 불구하고 범주형인 데이터도 있을 것이다. 
# 이런 친구들도 더미코딩을 해 줘야한다. 
# 이런 경우 우리들이 변수를 보고 수정을 해야하지만, 
# 시간이 없으니까 object 타입 열만 가져와서 해보자.
df = pd.get_dummies(
    df,
    # object 형태인 변수 다 가져옴
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )

# train / test 데이터셋으로 나누기(4번)
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

train_x = train_df.drop("SalePrice", axis=1)
train_y = train_df["SalePrice"]

test_x = test_df.drop("SalePrice", axis=1, errors='ignore')

#=========================================
# Validation(교차검증) 만들기(10번)
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

# valid_result 한 칸 계산하는 함수
def rmse(model):
    # -- -> + 로 만들어준 것
    score = np.sqrt(-cross_val_score(model, 
                                     train_x, 
                                     train_y, 
                                     cv = kf,
                                     n_jobs=-1, 
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

alpha_values = np.arange(0, 1000, 10)
# 막대기 만들어줌(valid_result)
mean_scores = np.zeros(len(alpha_values))

k=0
# for문으로 칸 다 채워서 막대기 채움
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

#==========================================
# 선형 회귀 모델 생성(8번)
model = Ridge(alpha=9.649999999999965) # valid는 lambda 알기 위해서 쓰는 것

# 모델 학습
model.fit(train_x, train_y) # train 적용

pred_y=model.predict(test_x) # test로 predict 하기

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("submission/sample_submission19.csv", index=False)
