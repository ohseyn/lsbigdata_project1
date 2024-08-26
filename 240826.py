import numpy as np

# 가로 벡터 * 세로 벡터
a = np.arange(1,4)
b = np.array([3, 6, 9]) # 3*x
a.dot(b)

# 행렬 * 벡터
a = np.array([1, 2, 3, 4]).reshape((2,2), order="F")
b = np.array([5, 6]).reshape(2,1)
a.dot(b) # a @ b

# 행렬 * 행렬
a = np.array([1, 2, 3, 4]).reshape((2,2), order="F")
b = np.array([5, 6, 7, 8]).reshape((2,2), order="F")
a @ b

# 행렬 * 행렬 예시1
a = np.array([1, 2, 1, 0, 2, 3]).reshape((2,3))
b = np.array([1, 0, -1, 1 ,2, 3]).reshape((3,2))
a @ b

# 행렬 * 행렬 예시2
a = np.array([3, 5, 7, 2, 4, 9, 3, 1, 0]).reshape((3, 3))
np.eye(3)

a @ np.eye(3)
np.eye(3) @ a

# 행렬 뒤집기(transpose)
a.transpose()
b = a[:, 0:2] # 행 다 가져오고, 열은 0,1 행만
b.transpose()

# 회귀분석 예측식
x=np.array([13, 15,
            12, 14,
            10, 11,
            5, 6]).reshape(4,2)

vec1=np.repeat(1,4).reshape(4,1)
matX=np.hstack((vec1, x)) # 옆으로 붙이기

beta_vec=np.array([2, 3, 1]).reshape(3,1)
# beta_vec=np.array([2, 0, 1]).reshape(3,1)
# beta_vec=np.array([2, -1, 1]).reshape(3,1)
matX @ beta_vec

y=np.array([20, 19, 20, 12]).reshape(4,1)
(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec)