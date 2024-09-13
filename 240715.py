import numpy as np
# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2024) # 랜덤값 보정
# 랜덤값 중에서 중복 되는 값 없이 출력하는 법
a = np.random.choice(np.arange(1, 21), 10, replace = False)
a = np.random.choice(np.arange(1, 4), 100, True, np.array([2/5, 2/5, 1/5]))
sum(a == 1)
sum(a == 2)
sum(a == 3)
a = np.random.randint(1, 21, 10) # 1에서 21까지 숫자 중 10개의 숫자 발생
print(a)
type(a) # numpy.ndarray
# 두 번째 값 추출
print(a[1])
a[::] #처음부터 끝까지 == a[::1]
a[:]
a[-2] # 맨 끝에서 두 번째
a[::2] # 처음부터 끝까지, 간격은 2
a[:2:] # a[:2]와 같다
a[1:6:2] #1부터 5까지, 간격은 2

# 1에서부터 1000 사이 3의 배수의 합은?
a = np.arange(1, 1001)
sum(a[2::3])

a = np.arange(3, 1001)
sum(a[::3])
print(a[[0, 2, 4]]) # 첫 번째, 세 번째, 다섯 번째 값 추출

# 두 번째 값 제외하고 추출
print(np.delete(a, 1))
np.delete(a, [1, 3])

# 인덱싱 중복 선택
print(a[[1, 1, 3, 2]])

b = a[a > 3] # 3보다 큰 벡터
print(b)
a > 3 # 논리 연산자(True, False 로 표시됨)

np.random.seed(2024) 
a = np.random.randint(1, 10000, 5) # 길이가 같아야 함
a[a >3000] = 3000
a
b = np.array(["A", "B", "C", "F", "W"])

a[[False, True, False, True, False]]
b[[False, True, False, True, False]]

b[(a > 2000) & (a < 5000)]
a[(a > 2000) & (a < 5000)] # a[조건을 만족하는 논리형 벡터]
(a > 2000) & (a < 5000)

# !pip install pydataset
import pydataset

df = pydataset.data('mtcars')
np_df = np.array(df['mpg'])

model_names = np.array(df.index)
model_names[(np_df < 15) & (np_df <= 20)]

# 15 이상 25 이하인 데이터 개수
sum((np_df >= 15) & (np_df <= 25))

# 15 작거나 22 이상인 데이터 개수
sum((np_df < 15) | (np_df >= 22))
model_names[(np_df < 15) | (np_df >= 22)]

#평균 mpg보다 높은(이상) 자동차 대수
sum(np_df >= np.mean(np_df))
model_names[np_df >= np.mean(np_df)]

#평균 mpg보다 낮은(미만) 자동차 대수
model_names[np_df < np.mean(np_df)]

np.random.seed(2024) 
a = np.random.randint(1, 100, 10)
a < 50
np.where(a < 50) # True인 인덱스 위치 반환

np.random.seed(2024) 
a = np.random.randint(1, 26346, 1000)
a
# 처음으로 5000보다 큰 숫자가 나오는 위치와 그 숫자는?
x = np.where(a>22000)
x
type(x) # 튜플(원소는 하나인 튜플)
x[0] 
type(x[0]) # numpy.ndarray
my_index # 위치
my_index = x[0][0]
a[my_index]
a[np.where(a>22000)][0]

# 처음으로 10000보다 큰 숫자 중 50번째 숫자 위치와 그 숫자: 81, 21052

np.random.seed(2024) 
a = np.random.randint(1, 26346, 1000)
x = np.where(a>10000)
x[0][49] # 위치
a[x[0][49]] # 숫자

# 500보다 작은 숫자들 중 가장 마지막으로 나오는 숫자 위치와 그 숫자: 960, 391

np.random.seed(2024) 
a = np.random.randint(1, 26346, 1000)
x = np.where(a<500)
x[0][-1]
a[x[0][-1]]

import numpy as np
a = np.array([20, np.nan, 13, 24, 309])
a
a+3
np.mean(a)
np.nanmean(a)
np.nan_to_num(a, nan = 0) # nan을 원하는 숫자로 변경
np.isnan(a) # array([False,  True, False, False, False])

a = None # 아무것도 출력 안 됨
b = np.nan # nan
b + 1 # nan
a + 1 # error

~np.isnan(a) # array([ True, False,  True,  True,  True])
a_filtered = a[~np.isnan(a)]
a_filtered # array([ 20.,  13.,  24., 309.]) nan 출력 안 됨

str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]] # array(['사과', '수박'], dtype='<U2')

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str) # 벡터는 한 가지 데이터 타입만 허용, 문자열 벡터 
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec)) # 리스트나 튜플이건 concatenate로 묶으면 numpy.array로 출력
combined_vec

col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16))) # row_stack
row_stacked

uneven_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 18)))
uneven_stacked

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2)) #  길이를 강제로 맞춰주고, 값을 앞에서부터 채움
vec1 # array([1, 2, 3, 4, 1, 2])
uneven_stacked = np.column_stack((vec1, vec2)) # array([[ 1, 12],
                                               #        [ 2, 13],
                                               #        [ 3, 14],
                                               #        [ 4, 15],
                                               #        [ 1, 16],
                                               #        [ 2, 17]])
uneven_stacked = np.vstack((vec1, vec2)) # array([[ 1,  2,  3,  4,  1,  2],
                                         #        [12, 13, 14, 15, 16, 17]])
uneven_stacked

# [21, 24, 31, 44, 58, 67]
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
a
b
x = np.empty(6)
x
x[1::2] = b # x[[1, 3, 5]] = b
x[0::2] = a # x[[0, 2, 4]] = a
x # 초기에는 2.4e+001 이렇게 표시되다가 통일되니 정수로 표시
