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