# lec6 행렬
import numpy as np

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5), np.arange(12, 16))) # numpy.ndarray
matrix = np.vstack((np.arange(1, 5), np.arange(12, 16)))
print("행렬:\n", matrix)

np.zeros(5)
np.zeros((5,4))
np.arange(1, 5).reshape((2,2))
# -1 통해서 크기를 자동으로 결정할 수 있음
np.arange(1, 7).reshape((2, -1)) 

# 0에서부터 99까지 수 중 랜덤하게 50개 숫자를 뽑아서 5 by 10 행렬을 만드세요(정수)
np.random.seed(2024)
mat_a = np.random.randint(0, 100, 50).reshape((5, 10), order = "F")
mat_a[2, 3]
mat_a[0:2, 3] # 행은 0, 1 두 개, 행은 3인 수들
mat_a[1:3, 1:4]
mat_a[3, ] # 행자리, 열자리 비어있는 경우 전체 행 또는 열 선택
mat_a[3,:]
mat_a[3,::2] 

mat_b = np.arange(1, 101).reshape((20, -1))
mat_b[1::2,:]
mat_b[[1, 4, 6, 14], ]

# 필터링
x = np.arange(1, 11).reshape((5, 2)) * 2
# 첫 번째 열에서 첫 번째, 두 번째, 다섯 번째 행의 원소 반환
filtered_elements = x[[True, True, False, False, True], 0]
# 디멘션이 2차원(행렬)에서 1차원(벡터)으로 변함
mat_b[:,1] # 1차원
mat_b[:,[1]] # 2차원
mat_b[:,1:2] # 2차원
mat_b[:,(1,)] # 2차원
mat_b[:,1].reshape(-1,1)
mat_b[mat_b[:,1]%7 == 0,:]

# 사진은 행렬
import matplotlib.pyplot as plt

# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)
# 행렬을 이미지로 표시
# 0에 가까울수록 검은색, 1에 가까울수록 흰색
plt.imshow(img1, cmap='gray', interpolation='nearest') # 숫자를 색으로 변환
plt.colorbar
plt.show()
plt.clf()

a = np.random.randint(0, 256, 20).reshape(4, -1)
a/255
plt.imshow(a/255, cmap='gray', interpolation='nearest') # 숫자를 색으로 변환
plt.colorbar
plt.show()
plt.clf()

import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

import imageio
# 이미지 읽기
jelly = imageio.imread("C:/Users/User/Documents/LSBigDataSchool/lsbigdata_project1/jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape) # (88, 50, 4) 88x50이 4장 겹쳐있음(3차원)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])
# 첫 3개의 채널은 빨강, 녹색, 파랑의 색깔 강도를 숫자로 표현, 마지막 채널은 투명도(opacity)

jelly[:, :, 0].shape #(88, 50)
jelly[:, :, 0].transpose().shape #(50, 88)

plt.imshow(jelly)
plt.imshow(jelly[:, :, 0].transpose())
plt.imshow(jelly[:, :, 0]) # R
plt.imshow(jelly[:, :, 1]) # G
plt.imshow(jelly[:, :, 2]) # B
plt.imshow(jelly[:, :, 3]) # 투명도
plt.axis('off') # 축 정보 없애기
plt.show()
plt.clf()

# 3차원 배열
# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

my_array = np.array([mat1, mat2])
my_array.shape # (2, 2, 3) 2행 3열이 2장

first_slice = my_array[0, :, :]
first_slice

filtered_slice = my_array[:, :, :-1] # 열 첫 번째에서 마지막 2번째까지만 출력(마지막 열 제외)

filtered_slice2 = my_array[:, :, [0,2]]
filtered_slice2 = my_array[:, :, ::2]

filtered_slice3 = my_array[:, 0, :] 
filtered_slice3.shape # 2차원 (2, 3)
filtered_slice3 = my_array[:, ::2, :] # 3차원 (2, 1, 3)
filtered_slice3.shape

filtered_slice4 = my_array[0, 1, [1,2]]
filtered_slice4 = my_array[0, 1, 1:3]

mat_x = np.arange(1, 101).reshape((5, 5, 4))
mat_y = np.arange(1, 101).reshape((10, 5, 2))
mat_y = np.arange(1, 101).reshape((-1, 5, 2))

my_array2 = np.array([my_array, my_array])
my_array2.shape # (2, 2, 2, 3) 2행 3열이 2장이 2번 겹쳐져 있다.
my_array2[0, :, :, :]
len(jelly)
jelly.shape # (88, 50, 4)

# numpy 배열 메서드
a = np.array([[1, 2, 3], [4, 5, 6]])
a.sum() # np.int64(21)
a.sum(axis=0) # array([5, 7, 9])
a.sum(axis=1) # array([ 6, 15])
a.mean() # np.float64(3.5)
a.mean(axis=0) # array([2.5, 3.5, 4.5])
a.mean(axis=1) # array([2., 5.])

mat_b = np.random.randint(0, 100, 50).reshape((5, -1))
mat_b.max() # 가장 큰 수
mat_b.max(axis=0) # 열별로 가장 큰 수
mat_b.max(axis=1) # 행별로 가장 큰 수

a = np.array([1, 3, 2, 5]).reshape((2,2))
a.cumsum() # 누적 합 array([ 1,  4,  6, 11])
a.cumprod() # 누적 곱 array([ 1,  3,  6, 30])

mat_b.cumsum(axis=1) # 사이즈 동일 
mat_b.cumprod(axis=1)
mat_b.reshape((2, 5, 5)) # 3차원(대괄호 개수가 차원의 개수)
mat_b.flatten() # 다차원을 하나로 펴줌(1차원)

d = np.array([1, 2, 3, 4, 5])
d.clip(2, 4) # 최소값 기준으로 최소값보다 작은 수는 최소값으로, 
             # 최대값 기준으로 최대값보다 큰 수는 최대값으로 바꿈.
d.tolist() # [1, 2, 3, 4, 5] 리스트로 됨(array가 빠짐)

# 균일확률변수 만들기
np.random.rand(1)
def X(num):
    return np.random.rand(num)

# X(1) X(1) X(1) / X(3) -> 독립적

# 베르누이 확률변수 모수: p 만들어보세요!
def Y(p):
    x = np.random.rand(1)
    return np.where(x < p, 1, 0)

def Y(num, p):
    x = np.random.rand(num)
    return np.where(x < p, 1, 0)

sum(Y(num=100, p=0.5))/100
Y(10000, 0.5).mean()

# 새로운 확률변수 가질 수 있는 값: 0, 1, 2
# 확률: 20%, 50%, 30%

def Z(num, p, q):
    x = np.random.rand(num)
    return np.where(x < p, 0, np.where(x < q, 1, 2))

Z(100, 0.2, 0.7).mean()
# 0.7(0.2 + 0.5)인 게 중요!

p = np.array([0.2, 0.5, 0.3])
def Z(p):
    x = np.random.rand(1)
    p_cumsum = p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where(x < p_cumsum[1], 1, 2))

Z(p)
