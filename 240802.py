import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 함수 y=x^2 +3 최소값이 나오는 입력값 구하기
def my_f(x):
    return x**2+3

my_f(3)

# 초기 추정값
initial_guess = [0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# z = x^2 + y^2 +3 최소값이 나오는 입력값 구하기
# 변수가 2개일 때 x라 해놓고, x 첫번째 값은 x, 두번째 값은 y
def my_f2(x):
    return x[0]**2+x[1]**2+3

my_f2([1,3])

# 초기 추정값
initial_guess = [1, 3]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# f(x, y, z) = (x-1)^2 + (y-2)^2 +(z-4)^ + 7 최소값과 최소값이 되는 자리
def my_f3(x):
    return ((x[0]-1)**2)+((x[1]-2)**2)+((x[2]-4)**2)+7

my_f3([3, 6, 5])

# 초기 추정값
initial_guess = [1, 2, 4]

# 최소값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

#=======================================
# 최소값을 찾을 다변수 함수 정의
def my_f(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 회귀직선 구하기

import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat)))

line_perform([36, 68])

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
