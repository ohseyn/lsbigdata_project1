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

# 12장
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

#=================================

# 선 그래프
economics = pd.read_csv("Data/economics.csv")
economics.info()
sns.lineplot(data = economics, x="date", y="unemploy")
plt.show()
plt.clf()

economics["date2"] = pd.to_datetime(economics["date"])
economics.info()
economics[["date", "date2"]]
economics["year"] = economics["date2"].dt.year # 어트리뷰트(변수가 지니고 있는 값)
economics["date2"].dt.month
economics["date2"].dt.day

economics["date2"].dt.month_name()
economics["quarter"] = economics["date2"].dt.quarter # 분기
economics[["date2", "quarter"]]
economics["date2"].dt.day_name() # 요일
economics["date2"] + pd.DateOffset(days=30) # 날짜 더하기
economics["date2"] + pd.DateOffset(months=1) # 날짜 더하기
sum(economics["date2"].dt.is_leap_year) # 윤년 체크

sns.lineplot(data = economics, x="year", y="unemploy", errorbar = None) # errorbar: 신뢰구간
sns.scatterplot(data = economics, x="year", y="unemploy", size=1)
plt.show()
plt.clf()

my_df = economics.groupby("year", as_index = False)\
        .agg(
            mean_year = ("unemploy", "mean"), 
            std_year = ("unemploy", "std"),
            n_year = ("unemploy", "count")
            )

# mean + 1.96*std/sqrt(n) (1.96으로 통일)
my_df["left_ci"] = my_df["mean_year"] - 1.96*my_df["std_year"]/np.sqrt(my_df["n_year"])
my_df["right_ci"] = my_df["mean_year"] + 1.96*my_df["std_year"]/np.sqrt(my_df["n_year"])

# x축:연도 y축:평균
x = my_df["year"]
y = my_df["mean_year"]
plt.scatter(x, my_df["left_ci"], color="red", s=1) # 신뢰구간
plt.scatter(x, my_df["right_ci"], color="green", s=1) # 신뢰구간
plt.plot(x, y)
plt.show()
plt.clf()
