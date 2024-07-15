#수학 함수
import math

# 제곱근
x = 4
math.sqrt(x)

# e의 지수
exp_val = math.exp(5)
print("e^5의 값은:", exp_val)

# 로그
log_val = math.log(10, 10)
print("10의 밑 10 로그 값은:", log_val)

# 팩토리얼
fact_val = math.factorial(5)
print("5의 팩토리얼은:", fact_val)

# 삼각함수
sin_val = math.sin(math.radians(90)) # 90도를 라디안으로 변환
print("90도의 사인 함수 값은:", sin_val)

cos_val = math.cos(math.radians(180))
print("180도의 코사인 함수 값은:", cos_val)

tan_val = math.tan(math.radians(45))
print("45도의 탄젠트 함수 값은:", tan_val)

# 정규분포 확률밀도함수
def normal_pdf(x, mu, sigma):
 sqrt_two_pi = math.sqrt(2 * math.pi)
 factor = 1 / (sigma * sqrt_two_pi)
 return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def my_normal_pdf(x, mu, sigma):
  #part1_1: 1/(sigma * math.sqrt(2 * math.pi))
  part_1: (sigma * math.sqrt(2 * math.pi))**(-1)
  part_2: math.exp((-(x-mu)**2) / (2*sigma**2)) # 제곱이 우선순위가 더 높음
  return part_1 * part_2

mu = 0
sigma = 1
x = 1

pdf_value = normal_pdf(x, mu, sigma)
print("정규분포 확률밀도함수 값은:", pdf_value)

# 복잡한 수식

def my_f(x, y, z):
  return ((x**2) + math.sqrt(y) + math.sin(z)) * math.exp(x)

my_f(2, 9, math.pi/2)

def my_g(x):
  return math.cos(x) + math.sin(x) * math.exp(x)

my_g(math.pi)

def fname(`indent('.') ? 'self' : ''`):
    """docstring for fname"""
    # TODO: write code...
   
# fcn -> Shift + Space
def frame(input):
    contents
    return

# pd + Shift + Space -> import pandas as pd
# np + Shift + Space -> import numpy as np
# Ctrl + Shift + c : 주석 처리리

import numpy as np

a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

a # array([1, 2, 3, 4, 5])
type(a) # numpy.ndarray
a[3] #np.int64(4) <- 정수(int)
a[2:] # 인덱싱해도 numpy.ndarray로 반환
a[1:4]

b = np.empty(3)
b # 0이랑 비슷한 수를 임의로 넣어둠
b[0] = 1
b[1] = 2
b[2] = 3
b
b[2] # np.float64(3.0) <- 실수(float)

vecl = np.array([1, 2, 3, 4, 5])
vecl = np.arange(100)
vecl - np.arange(1, 101, 0.5)
vecl

arr2 = np.arange(0, 2, 0.5) # array([0. , 0.5, 1. , 1.5])
arr2

linear_space1 = np.linspace(0, 1, 10) # arrayarray([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
                                      #             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])
linear_space1

linear_space1 = np.linspace(0, 100, 100)
linear_space1

linear_space2 = np.linspace(0, 1, 5, endpoint=False)
linear_space2

vec1 = np.arange(5)
vec1
np.repeat(vec1, 5)

vec2 = np.arange(0, -100, -1)
vec2

vec3 = np.arange(0, 100)
-vec3

np.tile(vec1, 3)
vec1 *2

vec1 + vec1
max(vec1)
sum(vec1)

# 35672 이하 홀수들의 합은?

vec4 = np.arange(1, 35673, 2) #마지막 숫자는 +1(종료값은 배열에 포함되지 않으므로)
vec4.sum()
sum(vec4)
sum(np.arange(1, 35673, 2))

b = np.array([[1, 2, 3], [4, 5, 6]])
b #array([[1, 2, 3],
  #      [4, 5, 6]])
length = len(b) 
shape = b.shape
size = b.size
length, shape, size

a = np.array([1, 2])
b = np.array([1, 2, 3, 4])
a+b

np.tile(a, 2) + b # tile: ([1, 2, 1, 2])
np.repeat(a, 2) + b # repeat: ([1, 1, 2, 2])
b == 3

# 10보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수는?
sum((np.arange(1, 10)%7) == 3) #True를 1로 계산하기 때문에
sum((np.arange(1, 35673)%7)==3)

a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b

a.shape # (3,)
b.shape # 숫자 하나라서 존재하지 않음

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
matrix.shape #(4, 3) 4행 3열
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
vector.shape #(3,)
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)
# 브로드캐스팅 결과:
# [[ 1.  2.  3.]
# [11. 12. 13.]
# [21. 22. 23.]
# [31. 32. 33.]]

vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1) # array([[1.],
                                                      #        [2.],
                                                      #        [3.],
                                                      #        [4.]])
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2) # array([[1., 2.],
                                                      #        [3., 4.]])
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4) # array([[1., 2., 3., 4.]])
vector
vector.shape #(4, 1)
vector = np.array([1.0, 2.0, 3.0, 4.0])
vector.shape #(4,)
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)
