from scipy.stats import bernoulli

# 확률질량함수 pmf: 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k, p)
bernoulli.pmf(1, 0.3)
bernoulli.pmf(0, 0.3)

import pandas as pd
house_df = pd.read_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/train.csv")
price_mean = house_df["SalePrice"].mean()
sub_df = pd.read_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission.csv")
sub_df["SalePrice"] = price_mean
sub_df.to_csv("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/sample_submission.csv", 
                index = False) # index = False 안 하면 열 하나가 추가 됨 
