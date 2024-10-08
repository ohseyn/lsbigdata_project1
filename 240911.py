# Stacking
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 데이터 로드(1번)
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
sub_df = pd.read_csv('sample_submission.csv')

# Nan 채우기
# 각 숫자 변수는 평균으로 채우기
# 각 범주형 변수는 최빈값으로 채우기
house_train.isna().sum()
house_test.isna().sum()

#==================================================
# house_train
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
    house_train[col].fillna("unknown", inplace=True)
house_train[Cate_selected].isna().sum()

#====================================================
# house_test
# 수치형만
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

# inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

# 범주형만
Categorical = house_test.select_dtypes(include = [object])
Categorical.isna().sum()
Cate_selected = Categorical.columns[Categorical.isna().sum() > 0]

for col in Cate_selected:
    # inplace=True 가 자동으로 업데이트 해줌(house_train[col] = 안 해줘도 됨)
    house_test[col].fillna("unknown", inplace=True)
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

# 이상치 탐색(6번)
train_df=train_df.query("GrLivArea <= 4500")

# x, y 나누기(7번)
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

test_x=test_df.drop("SalePrice", axis=1)

# 회귀 모델 생성(8번)
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

eln_model = ElasticNet()
# bootstrap
rf_model = RandomForestRegressor(n_estimators=100)

# 후보군
# alpha : 람다(패널티)
# l1_ratio : 알파(라쏘 가중치)
param_grid={
    "alpha": [0.1, 1.0, 10.0, 100.0],
    "l1_ratio": [0, 0.1, 0.5, 1.0]
}

# 그리드 별로 성능 평가
grid_search = GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error", # 성능평가
    cv=5
)

grid_search.fit(train_x, train_y)
grid_search.best_params_
best_eln_model=grid_search.best_estimator_

#==================================================

param_grid={
    "max_depth": [3, 5, 7],
    "min_samples_split": [20, 10, 5],
    "min_samples_leaf": [5, 10, 20, 30],
    "max_features": ["sqrt", "log2", None]
}

# 그리드 별로 성능 평가
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error", # 성능평가
    cv=5
)

grid_search.fit(train_x, train_y)
grid_search.best_params_
best_rf_model=grid_search.best_estimator_

# 스택킹
y1_hat=best_eln_model.predict(train_x)
y2_hat=best_rf_model.predict(train_x)

# 블렌더를 만들기 위한 학습 데이터
train_x_stack=pd.DataFrame({
    "y1": y1_hat,
    "y2": y2_hat
})

train_y[0]

from sklearn.linear_model import Ridge

rg_model = Ridge()

param_grid={
    "alpha": np.arange(0, 10, 0.01)
}

grid_search = GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error", # 성능평가
    cv=5
)

grid_search.fit(train_x_stack, train_y)
grid_search.best_params_
blander_model=grid_search.best_estimator_

# 엘라스틱넷 예측값에 0.36한 값/랜덤포레스트 예측값의 0.69
# coef는 블렌더 가중치
blander_model.coef_
blander_model.intercept_

pred_y_eln=best_eln_model.predict(test_x) # test 셋에 대한 집값
pred_y_rf=best_rf_model.predict(test_x) # test 셋에 대한 집값

test_x_stack=pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf
})

pred_y=blander_model.predict(test_x_stack)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y

# # csv 파일로 내보내기
sub_df.to_csv("submission/sample_submission21.csv", index=False)

#===================================================
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# 훈련용 데이터셋 만들기
# 1. 1000개의 데이터
train_x.shape[0]
# 0부터 1457까지의 배열 복원 추출
btstrap_index1=np.random.choice(np.arange(1458), 1000, replace=True)
bts_train_x1=train_x.iloc[btstrap_index1,:]
bts_train_y1=np.array(train_y)[btstrap_index1]

# 2. 1458개의 데이터
# 0부터 1457까지의 배열 복원 추출
btstrap_index2=np.random.choice(np.arange(1458), 1458, replace=True)
bts_train_x2=train_x.iloc[btstrap_index2,:]
bts_train_y2=np.array(train_y)[btstrap_index2]

# 회귀 분석을 위한 결정 트리 모델
model = DecisionTreeRegressor(random_state=42)
# 최대 깊이(max_depth), 최소 샘플 분할 수(min_samples_split) 하이퍼파라미터 그리드. 
# 각각 7부터 19까지, 10부터 29까지의 값을 탐색
param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}
# 파라미터 그리드를 사용하여 
# 교차 검증(cv=5)을 통해 최적의 하이퍼파라미터를 찾기
# 평가는 neg_mean_squared_error
grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(bts_train_x1, bts_train_y1)
grid_search.best_params_
# 최적의 파라미터로 학습된 모델
bts_model1=grid_search.best_estimator_

# 두 번째 bootstrapped 데이터로 모델을 다시 학습하고, 그 결과를 예측
grid_search.fit(bts_train_x2, bts_train_y2)
grid_search.best_params_
bts_model2=grid_search.best_estimator_

# 두 모델의 예측값을 평균내어 최종 결과를 생성
bts1_y=bts_model1.predict(test_x)
bts2_y=bts_model2.predict(test_x)
(bts1_y + bts2_y)/2

#=================================================
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# 여러 부트스트랩 샘플을 통해 각각 모델을 학습하고
# 그 예측값을 평균내거나 다수결을 통해 최종 결과 만듦

# 결정 트리 분류 모델을 사용하는 Bagging 모델
# 50개의 모델을 앙상블로 학습
# 각 모델은 100개의 샘플로 학습
bagging_model = BaggingClassifier(DecisionTreeClassifier(),
                                  n_estimators=50,
                                  max_samples=100, 
                                  n_jobs=-1, random_state=42)

# 결정 트리 회귀 모델을 사용하는 Bagging 모델
# 2개의 회귀 트리를 앙상블로 학습
bagging_model = BaggingRegressor(DecisionTreeRegressor(),
                                  n_estimators=2, 
                                  n_jobs=-1, random_state=42)

# * n_estimator: Bagging에 사용될 모델 개수
# * max_sample: 데이터셋 만들때 뽑을 표본크기

# bagging_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

# RandomForest: Bagging의 특수한 형태
# 여러 결정 트리를 학습하면서 트리 생성 시 랜덤성을 추가
# 각 트리가 독립적으로 자라게 함

# n_estimators: 학습할 트리의 개수
# max_leaf_node: 하나의 트리에서 가질 수 있는 최대 리프 노드 수
rf_model=RandomForestClassifier(n_estimators=50,
                                max_leaf_node=16,
                                n_jobs=-1, random_state=42)

# rf_model.fit(X_train, y_train)