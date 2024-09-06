import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# 펭귄 분류 문제
# y: 펭귄의 종류
# x1: bill_length_mm (부리 길이)
# x2: bill_depth_mm (부리 깊이)

df=penguins.dropna()
df=df[["species", "bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={
                    "species": "y",
                    'bill_length_mm': 'x1',
                    'bill_depth_mm': 'x2'})

# x1, x2 산점도를 그리되, 점 색깔은 종 별로 다르게!
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data = df, x = 'x1', y = 'x2', hue = "y")
# plt.axvline(x = 45, color="purple")
plt.axvline(x = 42.3, color="purple")

# 나누기 전 현재의 엔트로피?

p_i=df[["y"]].value_counts() / len(df["y"])
entropy_curr=-sum(p_i*np.log2(p_i))

# 45로 나눴을 때 엔트로피 평균?
# 데이터 포인트 개수
n1=df.query("x1 < 45").shape[0]
n2=df.query("x1 >= 45").shape[0]

# 어떤 종류로 예측?
y_hat1=df.query("x1 < 45")["y"].mode()
y_hat2=df.query("x1 >= 45")["y"].mode()

# 각 그룹 엔트로피는 얼마 인가요?
p_1=df.query("x1 < 45")["y"].value_counts()/len(df.query("x1 < 45")["y"])
entropy1=-sum(p_1*np.log2(p_1))
p_2= df.query("x1 >= 45")["y"].value_counts()/len(df.query("x1 >= 45")["y"])
entropy2=-sum(p_2*np.log2(p_2))

entropy1_x145 = (n1 * entropy1 + n2 * entropy2) / (n1+n2)

# 기준값 x를 넣으면 entropy값이 나오는 함수는?
# x1 기준으로 최적 기준값은 얼마인가?

def my_entropy(x):
    n1=df.query(f"x1<{x}").shape[0]
    n2=df.query(f"x1>={x}").shape[0]
    y_hat1=df.query(f"x1<{x}")["y"].mode()
    y_hat2=df.query(f"x1>={x}")["y"].mode()
    p_1=df.query(f"x1<{x}")["y"].value_counts()/len(df.query(f"x1<{x}")["y"])
    entropy1=-sum(p_1*np.log2(p_1))
    p_2= df.query(f"x1>={x}")["y"].value_counts()/len(df.query(f"x1>={x}")["y"])
    entropy2=-sum(p_2*np.log2(p_2))
    entropy = (n1 * entropy1 + n2 * entropy2) / (n1+n2)
    return entropy

# argmin을 써서 entropy 최소값을 찾아야 함

x1_min = df['x1'].min()
x1_max = df['x1'].max() 

input_x = np.arange(x1_min, x1_max, 0.1)
entropy_values = np.zeros(len(input_x))
for i, x in enumerate(input_x):
    entropy_values[i] = my_entropy(x)
optimal_entropy = input_x[np.argmin(entropy_values)]
entropy_values[102]

#===================================================

def entropy(col):
    entropy_i = []
    for i in range(len(df[col].unique())):
        df_left = df[df[col]< df[col].unique()[i]]
        df_right = df[df[col]>= df[col].unique()[i]]
        info_df_left = df_left[['y']].value_counts() / len(df_left)
        info_df_right = df_right[['y']].value_counts() / len(df_right)
        after_e_left = -sum(info_df_left*np.log2(info_df_left))
        after_e_right = -sum(info_df_right*np.log2(info_df_right))
        entropy_i.append(after_e_left * len(df_left)/len(df) + after_e_right * len(df_right)/len(df))
    return entropy_i

entropy_df = pd.DataFrame({
    'standard': df['x1'].unique(),
    'entropy' : entropy('x1')
    })

entropy_df.iloc[np.argmin(entropy_df['entropy']),:]

#===================================================
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

df_X=penguins.drop("species", axis=1)
df_X=df_X[["bill_length_mm", "bill_depth_mm"]]
y=penguins[["species"]] # 회귀분석모델보다 유연함

# 회귀는 MSE 씀(지금 종이라서 평균을 못 구함)
# 데이터 X, y 쪼개서 decisiontree에 넣고 depth가 2인 model fit

# 모델 생성
from sklearn.tree import DecisionTreeClassifier
## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42)

param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)

grid_search.fit(df_X,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

model = DecisionTreeClassifier(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(df_X,y)

from sklearn import tree
tree.plot_tree(model)

# 혼동 행렬
# 아델리: 'A'
# 친스트랩(아델리 아닌것): 'C'
from sklearn.metrics import confusion_matrix

y_true = np.array(['A', 'A', 'C', 'A', 'C', 'C', 'C'])
y_pred1 = np.array(['A', 'C', 'A', 'A', 'A', 'C', 'C'])
y_pred2 = np.array(['C', 'A', 'A', 'A', 'C', 'C', 'C'])

conf_mat=confusion_matrix(y_true=y_true, 
                          y_pred=y_pred1,
                          labels=["A", "C"])

conf_mat=confusion_matrix(y_true=y_true, 
                          y_pred=y_pred2,
                          labels=["A", "C"])

from sklearn.metrics import ConfusionMatrixDisplay

p=ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                         display_labels=("Adelie", "Chinstrap"))
p.plot(cmap="Blues")

