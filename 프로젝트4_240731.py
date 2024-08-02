# plot 2개
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

house_train = pd.read_csv('train.csv')

house_train1 = house_train[["SalePrice", "MoSold"]]
house_mo = house_train1.dropna(subset=["MoSold","SalePrice"])\
                    .groupby("MoSold", as_index = False)\
                    .agg(count = ("SalePrice","count"))\
                    .sort_values("MoSold", ascending = True)
sns.barplot(data=house_mo, x="MoSold", y="count", hue="MoSold")
plt.rcParams.update({"font.family":"Malgun Gothic"})
plt.xlabel("월(month)")
plt.ylabel("이사횟수(count)")
plt.tight_layout()
plt.show()
plt.clf()  

house_train2 = house_train[["YearBuilt", "OverallCond"]]

house_cond = house_train2.dropna(subset=["YearBuilt","OverallCond"])\
                    .groupby("OverallCond", as_index = False)\
                    .agg(count = ("YearBuilt", "count"))\
                    .sort_values("count", ascending = False)
sns.barplot(data = house_cond, x = "OverallCond", y = "count", hue = "OverallCond")
plt.tight_layout()
plt.show()
plt.clf()

house_train3 = house_train[["BldgType", "OverallCond"]]

house_bed = house_train3.dropna(subset=["BldgType","OverallCond"])\
                    .groupby(["OverallCond", "BldgType"], as_index = False)\
                    .agg(count = ("BldgType", "count"))\
                    .sort_values("count", ascending = False)
sns.barplot(data = house_bed, x = "OverallCond", y = "count", hue = "BldgType")
plt.tight_layout()
plt.show()
plt.clf()

house_train4 = house_train[["SalePrice", "Neighborhood"]]
house_nei = house_train4.dropna(subset=["SalePrice", "Neighborhood"])\
                    .groupby("Neighborhood", as_index=False)\
                    .agg(region_mean=("SalePrice", "mean"))\
                    .sort_values("region_mean", ascending=False)
            
sns.barplot(data=house_nei, x="Neighborhood", y="region_mean", hue="Neighborhood")
plt.title('Average SalePrice by Neighborhood')
plt.xticks(rotation=45, size=6)
plt.tight_layout()
plt.show()
plt.clf()
