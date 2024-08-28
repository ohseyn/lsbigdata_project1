import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
# 다항 특징 이름 생성
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
# X라는 데이터 프레임에 x 값을 넣고 열 이름을 x라 함
X = pd.DataFrame(x, columns=['x'])
# include_bias: beta0 * 1(이 부분을 False 해서 안 쓰겠다는 뜻)
# 다항 특징 20까지 생성
poly = PolynomialFeatures(degree=20, include_bias=False)
# 다항 특징 생성된 것을 X에 추가
X_poly = poly.fit_transform(X)
# 이를 데이터프레임으로 생성
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

# 주어진 모델에 대해 교차 검증을 수행
# root_mean_squared
def rmse(model):
    # -- -> + 로 만들어준 것
    # n_jobs: 병렬 처리를 지원하는 매개변수(CPU 코어 개수)
    score = np.sqrt(-cross_val_score(model, 
                                     X_poly, 
                                     y, 
                                     cv = kf,
                                     n_jobs=-1, 
                                     # 모든 CPU 코어를 사용하여 작업을 수행
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 알파 값(람다) 설정(1000열)
alpha_values = np.arange(0, 10, 0.01) # 테스트할 alpha 값의 범위
# 막대기 만들어줌(valid_result)
# zeros: 0으로 초기화
mean_scores = np.zeros(len(alpha_values)) # 각 alpha 값에 대한 RMSE 값을 저장하기 위한 공간

k=0
# for문으로 칸 다 채워서 막대기 채움
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    # lasso 돌린 값을 막대기 한 칸 한 칸 채워줌
    mean_scores[k] = rmse(lasso)
    k += 1

# 각 alpha 값과 해당 alpha 값에 대한 RMSE 값을 
# DataFrame으로 변환하여 저장
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

#============================================
# Lasso를 Ridge로 바꿈!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
X = pd.DataFrame(x, columns=['x'])
# include_bias: beta0 * 1(이 부분을 False 해서 안 쓰겠다는 뜻)
poly = PolynomialFeatures(degree=20, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

# 주어진 모델에 대해 교차 검증을 수행
# root_mean_squared
def rmse(model):
    # -- -> + 로 만들어준 것
    # n_jobs: 병렬 처리를 지원하는 매개변수(CPU 코어 개수)
    score = np.sqrt(-cross_val_score(model, 
                                     X_poly, 
                                     y, 
                                     cv = kf,
                                     n_jobs=-1, 
                                     # 모든 CPU 코어를 사용하여 작업을 수행
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 알파 값(람다) 설정
alpha_values = np.arange(0, 10, 0.01) # 테스트할 alpha 값의 범위
# 막대기 만들어줌(valid_result)
mean_scores = np.zeros(len(alpha_values)) # 각 alpha 값에 대한 RMSE 값을 저장하기 위한 공간
k=0
# for문으로 칸 다 채워서 막대기 채움
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    # lasso 돌린 값을 막대기 한 칸 한 칸 채워줌
    mean_scores[k] = rmse(ridge)
    k += 1

# 각 alpha 값과 해당 alpha 값에 대한 RMSE 값을 
# DataFrame으로 변환하여 저장
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
