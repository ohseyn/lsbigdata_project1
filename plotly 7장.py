!pip install palmerpenguins
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()
penguins.info()
penguins["species"].unique()
penguins.columns

# x: bill_length_mm
# y: bill_depth_mm

# trendline = "ols"
# Plotly Express에서 산점도에 선형 회귀선(linear regression line)을 추가하기 위한 옵션
fig=px.scatter(penguins, x= "bill_length_mm", y= "bill_depth_mm", color="species", trendline="ols")
# dict() = {}
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이와 깊이", font=dict(color="white", size=24)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(
        font=dict(color="white", size=14),  # 범례 폰트 크기 조정
        title=dict(text="펭귄 종", font=dict(color="white", size=14))  # 범례 제목 조정
    )
)
fig.update_traces(marker=dict(size=12, opacity=0.7)) # 점 크기 및 투명도 조정
fig.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins=penguins.dropna()
x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]

# 심슨의 역설(숨어있는 변수에 따라서 트랜드가 바뀜(분석이 바뀌게 됨))
model.fit(x, y)
linear_fit=model.predict(x)
model.coef_ # array([-0.08232675]) 
# 부리 길이가 1mm가 증가할때마다 부리 깊이가 0.08씩 줄어든다
model.intercept_ # np.float64(20.786648668433827)

fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀직선",
        line=dict(dash="dot", color="white")
    )
)
fig.show()

# 범주형 변수로 회귀분석 진행하기
# 더미 변수: 범주형 데이터("species")를 수치형 데이터로 변환하는 변수
# species 열의 모든 범주에 대해 더미 변수를 생성
penguins=penguins.dropna() # 결측치 제거
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False)
penguins_dummies.columns 
penguins_dummies.iloc[:,-3:]

# 3개 중 2개만 있어도 정보를 다 얻을 수 있음(drop_first = True)
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=True)

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

model = LinearRegression()
model.fit(x, y)

model.coef_ # array([ 0.20044313, -1.93307791, -5.10331533])
model.intercept_ # np.float64(10.565261622823762)

#       species     island  bill_length_mm  ...  body_mass_g     sex  year
#1       Adelie  Torgersen            39.5  ...       3800.0  female  2007
#340  Chinstrap      Dream            43.5  ...       3400.0  female  2009
# x1, x2, x3
# 39.5, 0, 0
# 43.5, 1, 0
# 0.2 * 43.5 + - 1.93 * True(1) + -5.1 * False + 10.56

# 회귀직선
# y = 0.2 * bill_length - 1.93 * species_Chinstrap - 5.1 * species_Gentoo + 10.56
# 0.2가 기울기, 종마다 y절편만 다름

regline_y=model.predict(x)

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=penguins["bill_length_mm"], y=y, 
                hue=penguins["species"], palette="deep",
                legend=False)
sns.scatterplot(x=penguins["bill_length_mm"], y=regline_y,
                color="black")
plt.show()
plt.clf()
