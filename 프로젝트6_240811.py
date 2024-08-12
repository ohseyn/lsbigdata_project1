import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

house = pd.read_csv('houseprice-with-lonlat.csv')
house["Sale_Price"].describe()

# 범위 설정
min_price = 0
max_price = 800000

plt.hist(house['Sale_Price'], bins=range(min_price, max_price, 10000))
plt.title('Histogram of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()
plt.clf()

#============================================

