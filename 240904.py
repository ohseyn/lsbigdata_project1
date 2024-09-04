import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

df=penguins.dropna()
df=df[["bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={"bill_length_mm": "y",
                    "bill_depth_mm": "x"})

# x=15 기준으로 나눴을 때, 데이터 포인트가 몇개씩 나눠지는지
n1=df.query("x < 15").shape[0] # sum(df["x"]<15) # 1번 그룹
n2=df.query("x >= 15").shape[0] # sum(df["x"]>=15) # 2번 그룹

# 1번 그룹, 2번 그룹 예측
y_hat1=df.query("x < 15").mean()[0]
y_hat2=df.query("x >= 15").mean()[0]

# 각 그룹 MSE
mse1=np.mean((df.query("x < 15")["y"]-y_hat1)**2)
mse2=np.mean((df.query("x >= 15")["y"]-y_hat2)**2)

# x=15의 MSE 가중평균
(mse1+mse2)/2
((mse1*n1)+(mse2*n2))/(len(df))

#=================================================
# x=20의 MSE 가중평균
n1=df.query("x < 20").shape[0] 
n2=df.query("x >= 20").shape[0]
y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]
mse1=np.mean((df.query("x < 20")["y"]-y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"]-y_hat2)**2)
(mse1+mse2)/2
((mse1*n1)+(mse2*n2))/(len(df))

#================================================
# 원래 MSE
np.mean((df["y"]-df["y"].mean())**2)