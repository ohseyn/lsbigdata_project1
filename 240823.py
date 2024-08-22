import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train=pd.read_csv("train.csv")
house_test=pd.read_csv("test.csv")
sub_df=pd.read_csv("sample_submission.csv")

house_train.shape
house_test.shape 
train_n=house_train.shape[0]

# 합친 이유는 한 번에 dummies 하려고
df=pd.concat([house_train, house_test], 
             axis=0, ignore_index=True)

# 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   판다스 시리즈
# house_train[["GrLivArea"]] 판다스 프레임

neighborhood_dummies = pd.get_dummies(
    df["Neighborhood"],
    drop_first=True
    )

# pd.concat([df_a, df_b], axis=1)
x= pd.concat([df[["GrLivArea", "GarageArea"]], 
             neighborhood_dummies], axis=1)
y = df["SalePrice"]

train_x=x.iloc[:train_n,]
test_x=x.iloc[train_n:,]
train_y=y[:train_n]

# Validation set(모의고사) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), 
                             size=438, replace=False)
valid_x=train_x.loc[val_index] # 30%
valid_y=train_y.loc[val_index] # 30%
train_x=train_x.drop(val_index) # 70%
train_y=train_y.drop(val_index) # 70%

# 이상치 제거
train_x=train_x.query("GrLivArea <= 4500")

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 성능 측정
y_hat=model.predict(valid_x) # valid_y에 대한 추정값
np.sqrt(np.mean((valid_y-y_hat)**2))
