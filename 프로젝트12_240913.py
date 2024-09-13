# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# 범주형 변수를 원-핫 인코딩
from sklearn.preprocessing import OneHotEncoder
# 다양한 전처리를 데이터의 특정 칼럼에 적용
from sklearn.compose import ColumnTransformer

# 데이터 로드
spaceship_train = pd.read_csv('Spaceship/train.csv')
spaceship_test = pd.read_csv('Spaceship/test.csv')
sub_df = pd.read_csv('Spaceship/sample_submission.csv')

spaceship_train.info()
spaceship_test.info()

# 데이터 합치기
all_data=pd.concat([spaceship_train, spaceship_test])
all_data=all_data.drop(['PassengerId', 'Name'], axis=1)

# 범주형 칼럼(데이터 타입이 object인 칼럼)을 c 변수에 저장
# 마지막 칼럼 Trasnported 제외
c = all_data.columns[all_data.dtypes == object][:-1]

# LabelEncoder를 사용하여 범주형 데이터를 숫자형으로 변환
from sklearn.preprocessing import LabelEncoder
# 인스턴스 생성
le = LabelEncoder()

# 숫자로 바꿈: 머신러닝 모델이 데이터를 이해하고 처리하기 위해서는 
# 모든 데이터를 숫자로 바꾸어야 함

# 원핫인코딩
# 범주형 데이터가 숫자로 바꿨을 때,
# 순서나 크기와 관계없다는 것을 모델에게 명확히 알려주는 방법
for i in c:
    all_data[i] = le.fit_transform(all_data[i])

# 결측치 처리
# inplace=True는 원본 데이터프레임을 직접 수정
all_data.fillna(-1, inplace=True)

# train / test 데이터셋
train_n=len(spaceship_train)
train=all_data.iloc[:train_n,]
test=all_data.iloc[train_n:,]

# train 데이터에서 타겟 변수(Transported)를 분리하여 y에 저장
y = train['Transported'].astype("bool")

# OneHotEncoder 설정
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first',
                              handle_unknown='ignore'), c)
    ], remainder='passthrough')

# 학습 데이터와 테스트 데이터에 전처리 적용
X_train = preprocessor.fit_transform(train.drop(['Transported'], axis=1))
X_test = preprocessor.transform(test)

# 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y)

# 예측 수행
predictions = model.predict(X_test)

# Transported 바꾸기
sub_df["Transported"] = predictions

# csv 파일로 내보내기
sub_df.to_csv("submission/spaceship_sample_submission_1.csv", index=False)

#==================================================
# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# 범주형 변수를 원-핫 인코딩
from sklearn.preprocessing import OneHotEncoder
# 데이터의 다양한 칼럼을 전처리
from sklearn.compose import ColumnTransformer

# 데이터 로드
spaceship_train = pd.read_csv('Spaceship/train.csv')
spaceship_test = pd.read_csv('Spaceship/test.csv')
sub_df = pd.read_csv('Spaceship/sample_submission.csv')

spaceship_train.info()
spaceship_test.info()

all_data=pd.concat([spaceship_train, spaceship_test])
all_data=all_data.drop(['PassengerId', 'Name'], axis=1)

# 범주형 칼럼
c = all_data.columns[all_data.dtypes == object][:-1]

# 정수형 전처리
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# 숫자로 바꿈: 머신러닝 모델이 데이터를 이해하고 처리하기 위해서는 
# 모든 데이터를 숫자로 바꾸어야 함

# 원핫인코딩
# 범주형 데이터가 숫자로 바꿨을 때,
# 순서나 크기와 관계없다는 것을 모델에게 명확히 알려주는 방법
for i in c:
    all_data[i] = le.fit_transform(all_data[i])

# 결측치 처리
all_data.fillna(-1, inplace=True)

# train / test 데이터셋
train_n=len(spaceship_train)
train=all_data.iloc[:train_n,]
test=all_data.iloc[train_n:,]

# 타겟 변수 분리
y = train['Transported'].astype("bool")

# OneHotEncoder 설정
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first',
                              handle_unknown='ignore'), c)
    ], remainder='passthrough')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# 파이프라인 설정
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('model', RandomForestClassifier(random_state=42))
])
# 하이퍼파라미터 튜닝
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(train.drop(['Transported'], axis=1), y)

# 최적 모델
best_model = grid_search.best_estimator_

# 예측 수행
predictions = best_model.predict(test)

# Transported 바꾸기
sub_df["Transported"] = predictions

# csv 파일로 내보내기
sub_df.to_csv("submission/spaceship_sample_submission_3.csv", index=False)
