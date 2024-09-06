# ElasticNet, GridSearchCV
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

# Validation 셋(모의고사 셋) 만들기(5번)
np.random.seed(42)
val_index=np.random.choice(np.arange(train_n), size=438, replace=False)

# train => valid / train 데이터셋(5번)
valid_df=train_df.loc[val_index]  # 30%
train_df=train_df.drop(val_index) # 70%

# 이상치 탐색(6번)
train_df=train_df.query("GrLivArea <= 4500")

# x, y 나누기(7번)
# Neighborhood_: Neighborhood_로 시작하는 열 선택
# regex (Regular Expression, 정규방정식)
# ^: 시작, $: 끝남, |: or
# 선택된 열들 train_x에 저장하여 학습 데이터의 입력 변수로 사용
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

valid_x=valid_df.drop("SalePrice", axis=1)
valid_y=valid_df["SalePrice"]

test_x=test_df.drop("SalePrice", axis=1)

# 회귀 모델 생성(8번)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.03)
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.03)
model = LinearRegression()
from sklearn.linear_model import ElasticNet
model = ElasticNet()

# 후보군
# alpha : 람다(패널티)
# l1_ratio : 알파(라쏘 가중치)
param_grid={
    "alpha": [0.1, 1.0, 10.0, 100.0],
    "l1_ratio": [0, 0.1, 0.5, 1.0]
}

# alpha 하나, l1_atio=1 로 설정하면 라쏘가 됨
from sklearn.model_selection import GridSearchCV

# 그리드 별로 성능 평가
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error", # 성능평가
    cv=5
)

grid_search.fit(train_x, train_y)
grid_search.best_params_
grid_search.cv_results_
# neg_mean_squared_error로 해서 마이너스 값 나옴
grid_search.best_score_ # 성능(best_params에 해당하는 성능)
best_model=grid_search.best_estimator_

best_model.predict(valid_x)

# 모델 학습
model.fit(train_x, train_y)

# 성능 측정(9번)
y_hat = model.predict(valid_x)
# 강사님은 이걸로 해서 값이 높게 나온 것
# np.sqrt(np.mean((valid_y-y_hat)**2))
np.mean(np.sqrt((valid_y-y_hat)**2))

pred_y=model.predict(test_x) # test 셋에 대한 집값

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("submission/sample_submission20.csv", index=False)

#=================================================
# 트리 시각화
# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수: bill_length_mm
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import OneHotEncoder

penguins = load_penguins()
penguins.head()

# Nan 채우기
quantitative = penguins.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    penguins[col].fillna(penguins[col].mean(), inplace=True)
penguins[quant_selected].isna().sum()

# 범주형 채우기
Categorical = penguins.select_dtypes(include = [object])
Categorical.isna().sum()
Cate_selected = Categorical.columns[Categorical.isna().sum() > 0]

for col in Cate_selected:
    penguins[col].fillna(penguins[col].mode()[0], inplace=True)
penguins[Cate_selected].isna().sum()

df = penguins
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
)

x=df.drop("bill_length_mm", axis=1)
y=df[['bill_length_mm']]

# 모델 생성
from sklearn.linear_model import ElasticNet
model = ElasticNet()

# 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

param_grid={
    'alpha': np.arange(0, 0.2, 0.01),
    'l1_ratio': np.arange(0.8, 1, 0.01)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(x,y)

grid_search.best_params_ #alpha=0.19, l1_ratio=0.99
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

# 모델 생성
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}

# 하이퍼파라미터 튜닝
grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(x,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

model = DecisionTreeRegressor(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(x,y)

from sklearn import tree
tree.plot_tree(model)

#=============================================
# Entropy
# 빨2/파3
p_r=2/5
p_b=3/5
h_zero=-p_r*np.log2(p_r)-p_b*np.log2(p_b)

# 빨1/파3
p_r=1/4
p_b=3/4
h_zero=-p_r*np.log2(p_r)-p_b*np.log2(p_b)
