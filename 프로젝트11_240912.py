# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# 데이터 로드
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
sub_df = pd.read_csv('sample_submission.csv')

# NaN 채우기
# 수치형 변수는 평균으로, 범주형 변수는 'unknown'으로 채우기
for df in [house_train, house_test]:
    # 수치형 변수 처리
    quantitative = df.select_dtypes(include=[int, float])
    for col in quantitative.columns:
        df[col].fillna(df[col].mean(), inplace=True)
    
    # 범주형 변수 처리
    categorical = df.select_dtypes(include=[object])
    for col in categorical.columns:
        df[col].fillna('unknown', inplace=True)

# 데이터 통합 및 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)
df = pd.get_dummies(df, drop_first=True)

# 훈련/테스트 데이터셋 나누기
train_n = len(house_train)
train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]

# 이상치 제거
train_df = train_df.query("GrLivArea <= 4500")

# 특성과 목표 변수 나누기
train_x = train_df.drop("SalePrice", axis=1)
train_y = train_df["SalePrice"]
test_x = test_df.drop("SalePrice", axis=1)

# 데이터 스케일링
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# ElasticNet 모델과 하이퍼파라미터 튜닝
eln_model = ElasticNet()
param_grid_eln = {
    "alpha": np.arange(65.0, 66.0, 0.1),
    "l1_ratio": [0, 0.1, 0.5, 1.0]
}
grid_search_eln = GridSearchCV(eln_model, param_grid_eln, scoring="neg_mean_squared_error", cv=5)
grid_search_eln.fit(train_x_scaled, train_y)
best_eln_model = grid_search_eln.best_estimator_

# RandomForestRegressor 모델과 하이퍼파라미터 튜닝
rf_model = RandomForestRegressor(n_estimators=100)
param_grid_rf = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [10, 5],
    "min_samples_leaf": [5, 10, 20, 30],
    "max_features": ["sqrt", "log2", None]
}
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, scoring="neg_mean_squared_error", cv=5)
grid_search_rf.fit(train_x_scaled, train_y)
best_rf_model = grid_search_rf.best_estimator_

# 스택킹: 예측 값 계산
y1_hat_scaled = best_eln_model.predict(train_x_scaled)
y2_hat_scaled = best_rf_model.predict(train_x_scaled)

# 블렌더 학습 데이터 생성
train_x_stack_scaled = pd.DataFrame({
    "y1": y1_hat_scaled,
    "y2": y2_hat_scaled
})

# Ridge 회귀로 블렌더 모델 학습
rg_model = Ridge()
param_grid_rg = {"alpha": np.arange(0, 10, 0.01)}
grid_search_rg = GridSearchCV(rg_model, param_grid_rg, scoring="neg_mean_squared_error", cv=5)
grid_search_rg.fit(train_x_stack_scaled, train_y)
blander_model = grid_search_rg.best_estimator_

# 테스트 데이터에 대한 예측
pred_y_eln_scaled = best_eln_model.predict(test_x_scaled)
pred_y_rf_scaled = best_rf_model.predict(test_x_scaled)
test_x_stack_scaled = pd.DataFrame({
    'y1': pred_y_eln_scaled,
    'y2': pred_y_rf_scaled
})

# 최종 예측
pred_y = blander_model.predict(test_x_stack_scaled)

# 결과를 파일로 저장
sub_df["SalePrice"] = pred_y
sub_df.to_csv("submission/sample_submission24.csv", index=False)