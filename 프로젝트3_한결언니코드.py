import pandas as pd
import numpy as np

house_train = pd.read_csv('./data/houseprice/train.csv')
house_train.info()
house_train = house_train[["Id", "OverallQual","TotRmsAbvGrd", "SalePrice"]]
house_train.info()

# OverallQual 평균
o_house_mean = house_train.groupby("OverallQual", as_index=False)\
           .agg(mean_o = ("SalePrice", "mean"))
o_house_mean

# TotRmsAbvGrd 평균
t_house_mean = house_train.groupby("TotRmsAbvGrd", as_index=False)\
            .agg(mean_t = ("SalePrice", "mean"))
t_house_mean

# OverallQual 평균 시각화
import matplotlib.pyplot as plt
x = o_house_mean["OverallQual"]
y = o_house_mean['mean_o']
plt.plot(x, y)
plt.show()
plt.clf()

# TotRmsAbvGrd 평균 시각화
import matplotlib.pyplot as plt
x = t_house_mean["TotRmsAbvGrd"]
y = t_house_mean['mean_t']
plt.plot(x, y)
plt.show()
plt.clf()

# test 데이터 불러오기
house_test = pd.read_csv('./data/houseprice/test.csv')
house_test = house_test[["Id", "OverallQual"]]
house_test

# merge
house_test = pd.merge(house_test, o_house_mean, how="left", on = "OverallQual")
house_test = house_test.rename(columns={'mean_o':"SalePrice"})
house_test

sum(house_test["SalePrice"].isna())

# sub 데이터 불러오기
sub_df = pd.read_csv('./data/houseprice/sample_submission.csv')
sub_df

# SalePrice 바꿔치기
sub_df['SalePrice'] = house_test['SalePrice']
sub_df

sub_df.to_csv('./data/houseprice/sample_submission3.csv', index=False)
