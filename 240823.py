import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 데이터 로드(1번)
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
sub_df = pd.read_csv('sample_submission.csv')

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩(2,3번번)
df = pd.concat([house_train, house_test], ignore_index=True)
df = pd.get_dummies(
    df,
    columns= ["Neighborhood"],
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
selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns
# 선택된 열들 train_x에 저장하여 학습 데이터의 입력 변수로 사용
train_x=train_df[selected_columns]
train_y=train_df["SalePrice"]

valid_x=valid_df[selected_columns]
valid_y=valid_df["SalePrice"]

test_x=test_df[selected_columns]

# 선형 회귀 모델 생성(8번)
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)

# 성능 측정(9번)
y_hat = model.predict(valid_x)
np.mean(np.sqrt((valid_y-y_hat)**2))

# test 셋 결측치 채우기
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("submission/sample_submission13.csv", index=False)

#==========================================
# 범주형, 숫자형 모두 다 입력값으로!
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

# 선형 회귀 모델 생성(8번)
model = LinearRegression()

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
sub_df.to_csv("submission/sample_submission14.csv", index=False)
