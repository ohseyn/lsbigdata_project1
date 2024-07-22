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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

mpg = pd.read_csv("C:/Data/mpg.csv")
sns.scatterplot(data = mpg, x = "cty", y = "hwy")
plt.show()
plt.clf()

midwest = pd.read_csv("C:/Data/midwest.csv")
sns.scatterplot(data = midwest, x = "poptotal", y = "popasian").set(xlim=[0,500000], ylim=[0,10000])
plt.show()
plt.clf()

mpg = pd.read_csv("C:/Data/mpg.csv")
suv = mpg.query("category == 'suv'").groupby("manufacturer", as_index = False)\
                                    .agg(mean_cty = ("cty", "mean"))\
                                    .sort_values("mean_cty", ascending = False)\
                                    .head(5)
sns.barplot(data = suv, x = "manufacturer", y = "mean_cty", hue = "manufacturer")
plt.show()
plt.clf()

df_category = mpg.groupby("category", as_index = False)\
                .agg(category_count = ("category", "count"))\
                .sort_values("category_count", ascending = False)
sns.barplot(data = df_category, x = "category", y = "category_count", hue = "category")
plt.show()
plt.clf()
