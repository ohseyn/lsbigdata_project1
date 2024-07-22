import pandas as pd

mpg = pd.read_csv("Data/mpg.csv")
mpg.shape

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=mpg, x="displ", y="hwy")
sns.scatterplot(data=mpg, x="displ", y="hwy").set(xlim=[3,6], ylim=[10,30])
sns.scatterplot(data=mpg, x="displ", y="hwy", hue="drv")
plt.show()
plt.clf()

# 막대 그래프
mpg["drv"].unique()
df_mpg = mpg.groupby("drv", as_index=False).agg(mean_hwy = ("hwy", "mean"))
sns.barplot(data=df_mpg, x="drv", y="mean_hwy", hue="drv")
sns.barplot(data=df_mpg.sort_values("mean_hwy", ascending = False), x="drv", y="mean_hwy", hue="drv")
plt.show()
plt.clf()

# 빈도 막대 그래프
df_mpg = mpg.groupby("drv", as_index=False).agg(n=("drv", "count"))
sns.barplot(data=df_mpg, x="drv", y="n", hue="drv") # barplot(막대), 3행 2열 데이터
sns.countplot(data=mpg, x="drv", hue="drv") # countplot(빈도 막대), 원본 데이터
plt.show()
plt.clf()

# 산점도 만들기
!pip install plotly
import plotly.express as px
fig = px.scatter(data_frame = mpg, x = "cty", y = "hwy", color="drv")
fig.show()
fig.clf()

# 막대 그래프 만들기
df = mpg.groupby("category", as_index = False).agg(n = ("category", "count"))
fig = px.bar(data_frame = df, x = "category", y = "n", color = "category")
fig.show()
fig.clf()
