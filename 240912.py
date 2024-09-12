import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import boxcox

# 예제 데이터 생성 (양수 데이터)
np.random.seed(0)
# 지수 분포를 따르는 난수 1000개 생성
# scale: 지수 분포의 스케일 파라미터
# 스케일 파라미터: 지수 분포 평균
# 스케일^2: 분산
# 스케일 파라미터가 클수록 지수 분포의 꼬리가 더 길어지고, 값이 커질 확률이 높아짐
# 즉, 데이터의 평균 대기 시간(간격) 길어짐
# 반대로 작으면, 지수 분포의 꼬리가 짧아지고, 값이 작을 확률이 높아짐
# 즉, 데이터의 평균 대기 시간이 짧아진다.
data = np.random.exponential(scale=2, size=1000)

# subplot으로 boxcox 되기 전/후를 나란히 배치
# Box-Cox 변환 전 데이터의 히스토그램
plt.figure(figsize=(12, 6))
# 1행 2열의 서브플롯 중 첫 번째 위치에 현재의 플롯을 설정
plt.subplot(1, 2, 1)
# 30개의 구간으로 데이터를 나눔
# alpha: 색상의 투명도
plt.hist(data, bins=30, color='blue', alpha=0.7)
plt.title('Before Box-Cox Transformation')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Box-Cox 변환
# boxcox된 데이터, 최적의 람다값 반환
data_boxcox, best_lambda = boxcox(data)

# Box-Cox 변환 후 데이터의 히스토그램
# 1행 2열의 서브플롯 중 두 번째 위치에 현재의 플롯을 설정
plt.subplot(1, 2, 2) 
plt.hist(data_boxcox, bins=30, color='green', alpha=0.7)
plt.title('After Box-Cox Transformation')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 서브플롯 간의 간격을 자동으로 조정
plt.tight_layout()

# 최적의 람다 값 출력
print(f'Box-Cox transformation에 대한 최적의 람다값: {best_lambda}')

#==========================================================
# GBRT
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
# 0과 1 사이의 균등 분포에서 난수 100개를 생성
# 0.5를 빼서 범위를 -0.5에서 0.5로 만들기
X = np.random.rand(100) - 0.5
# X의 제곱에 3을 곱하고, 작은 난수를 추가하여 y를 생성
y = 3 * X**2 + 0.05*np.random.randn(100)
# 100행 1열의 2D 배열로 변형
# Scikit-learn의 회귀 모델은 2D 배열의 입력
X=X.reshape(100,1)

# 원래 데이터의 분포 시각화
import matplotlib.pyplot as plt
plt.scatter(x=X, y=y)

# 깊이가 2인 결정 트리 회귀 모델을 생성
tree_model1=DecisionTreeRegressor(max_depth=2)
# tree_model1 모델을 학습
tree_model1.fit(X, y)
# 학습된 모델을 사용하여 X에 대한 예측값을 계산
y_tree1=tree_model1.predict(X)

# 1차 트리 예측값 시각화
import matplotlib.pyplot as plt
plt.scatter(x=X, y=y)
# 첫 번째 결정 트리 모델의 예측값을 산점도로 추가로 시각화
# 모델의 예측 성능을 확인
plt.scatter(x=X, y=y_tree1)

# 첫 번째 결정 트리 모델의 예측값을 원본 y에서 빼서 잔차(y2)를 계산
# 모델이 아직 설명하지 못한 부분
y2=y-tree_model1.predict(X)
# 깊이가 2인 결정 트리 회귀 모델을 생성
tree_model2=DecisionTreeRegressor(max_depth=2)
# 잔차 y2를 사용하여 두 번째 결정 트리 모델을 학습
tree_model2.fit(X, y2)
# 두 번째 모델을 사용하여 X에 대한 예측값을 계산
y_tree2=tree_model2.predict(X)

# 두번째 y 데이터 (y-1차 트리 예측값)
plt.scatter(x=X, y=y2)
# 두 번째 모델이 잔차를 얼마나 잘 설명하는지 보여줌
plt.scatter(x=X, y=y_tree2)

# 1차 + 2차 트리 예측값 시각화
plt.scatter(x=X, y=y)
plt.scatter(x=X, y=y_tree1+y_tree2)

# 두 번째 결정 트리 모델의 예측값을 잔차 y2에서 빼서 세 번째 잔차 y3를 계산
y3=y2-tree_model2.predict(X)
tree_model3=DecisionTreeRegressor(max_depth=2)
tree_model3.fit(X, y3)
y_tree3=tree_model3.predict(X)

# 세번째 y 데이터 (y2-2차 트리 예측값)
plt.scatter(x=X, y=y3)
# 세 번째 모델이 잔차를 얼마나 잘 설명하는지 보여줌
plt.scatter(x=X, y=y_tree3)

# 1차 + 2차 + 3차 트리 예측값 시각화
# 학습률(learning rate)을 0.1로 설정
l_rate=0.1
plt.scatter(x=X, y=y)
# 첫 번째, 두 번째, 세 번째 결정 트리 모델의 예측값을 
# 학습률 l_rate를 적용하여 합산한 값
# 세 모델의 예측값을 조합하여 원본 데이터 y에 얼마나 근접하는지를 보여줌
plt.scatter(x=X, y=y_tree1+l_rate*y_tree2+l_rate*y_tree3)

# 예측할 새로운 데이터 포인트를 생성
X_new=np.array([[0.5]])

# 세 결정 트리 모델을 사용하여 새로운 데이터 포인트에 대한 예측값을 계산하고 합산
tree_model1.predict(X_new) + tree_model2.predict(X_new) + tree_model3.predict(X_new)

# 위 내용을 scikit-learn을 사용해서 구현
from sklearn.ensemble import GradientBoostingRegressor

# 깊이가 2인 결정 트리를 기본 모델로 사용 
# 3개의 트리를 사용하는 그래디언트 부스팅 회귀 모델을 생성. 
# 학습률은 1.0으로 설정
gbrt=GradientBoostingRegressor(max_depth=2,
                               n_estimators=3,
                               learning_rate=1.0,
                               random_state=42)

# 그래디언트 부스팅 모델을 학습
gbrt.fit(X,y)
# X에 대한 예측값
gbrt.predict(X)

#================================================
class CookieMaker:
    # 생성자
    # CookieMaker 객체가 생성될 때 자동으로 호출
    def __init__(self):
        # result(인스턴스 변수)를 0으로 초기화
        self.result = 0

    # 함수들
    def add(self, num):
        self.result += num
        return self.result
    
    def reset(self):
        self.result = 0
        return self.result

# CookieMaker 클래스의 인스턴스(객체)를 생성
# cookie라는 변수에 할당
cookie=CookieMaker()
cookie.result

cookie.add(3)
cookie.result

# 클래스 메서드가 아닌 인스턴스 메서드를 클래스 이름으로 호출하는 예
CookieMaker.add(cookie, 4)
cookie.result

cookie.reset()
cookie.result