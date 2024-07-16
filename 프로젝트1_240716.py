import pandas as pd
import numpy as np

data = pd.read_csv('Data/시도별_화재발생_현황_총괄.csv')
data

data_2020 = data[['행정구역별'] + data.filter(like='2020').columns.tolist()]
data_2020

data_2020.columns = data_2020.iloc[0]
data_2020 = data_2020[1:]
data_2020 = data_2020.reset_index(drop=True)
data_2020

data_2020_pop = data_2020.iloc[:, 0:4]
data_2020_pop

# 문자형을 숫자형으로 변환
data_2020_pop['건수 (건)'] = pd.to_numeric(data_2020_pop['건수 (건)'])
data_2020_pop['사망 (명)'] = pd.to_numeric(data_2020_pop['사망 (명)'])
data_2020_pop['부상 (명)'] = pd.to_numeric(data_2020_pop['부상 (명)'])

# 변수명 변경
data_2020_pop = data_2020_pop.rename(columns = {"건수 (건)" : "건수"})
data_2020_pop = data_2020_pop.rename(columns = {"사망 (명)" : "사망자수"})
data_2020_pop = data_2020_pop.rename(columns = {"부상 (명)" : "부상자수"})

# 건수별로 정렬 
data_2020_pop.sort_values("건수", ascending = False)

# 인명 피해
data_2020_pop["total"] = data_2020_pop["사망자수"] + data_2020_pop["부상자수"]
data_2020_pop.head()

# 위험도 추가
count_mean = 38659/17  #평균
data_2020_pop["위험도"] = np.where(data_2020_pop["건수"] >= count_mean, "dan", "saf")
data_2020_pop.head()

# 빈도 막대 그래프
data_2020_pop["위험도"].value_counts()
data_2020_pop["위험도"].value_counts().plot.bar(rot=0) #rot=0 : 축 이름 수평
plt.show()
plt.clf()

# 시도별 인명피해 그래프
data_2020_pop["total"].plot.bar(rot = 0)
plt.show()
