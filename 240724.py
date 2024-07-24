from scipy.stats import bernoulli

# 확률질량함수 pmf: 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k, p)
bernoulli.pmf(1, 0.3)
bernoulli.pmf(0, 0.3)

import pandas as pd
df_house = pd.read_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/train.csv")
price_mean = df_house["SalePrice"].mean()
