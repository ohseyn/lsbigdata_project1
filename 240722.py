import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 리스트 예제
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]

# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()

# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))
range_list[3] = "LS 빅데이터 스쿨"
range_list[2] = "ohseyn"
range_list[1] = ["1st", "2nd", "3rd"]
range_list[1][2]

# 리스트 내포(comprehension)
# 1. 대괄호로 쌓여져 있다. -> 리스트
# 2. 넣고 싶은 수식 표현을 x를 사용해서 표현
# 3. for ... in ... 을 사용해서 원소 정보 제공
list(range(10))
squares = [x**2 for x in range(10)]
squares = [x**2 for x in [3, 5, 2, 15]]
squares = [x**2 for x in np.array([3, 5, 2, 15])]
exam = pd.read_csv("Data/exam.csv")
squares = [x**2 for x in exam["math"]]

# 리스트 연결
3+2
"안녕" + "하세요"
"안녕"*3
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2

(list1*3) + (list2*5)

# 리스트 각 원소별 반복
numbers = [5, 2, 3]
repeated_list = [x for x in numbers for _ in range(3)]
repeated_list = [x for x in numbers for y in range(3)]

# _의 의미: 앞에 나온 값을 가리킴
5 + 4
_ + 6 # 여기서 _는 9를 의미

# 값 생략, 자리 차지(placeholder)
a, _, b = (1, 2, 4)
a; b 
_

# for 루프 문법
# for i in 범위:
#   작동방식
for x in [4, 1, 2, 3]:
    print(x)

for i in range(5):
    print(i**2)

# 리스트를 하나 만들어서 for 루프를 사용하여 2, 4, 6, 8 ... 20까지 수를 채워넣기
[x for x in range(2, 21, 2)]

mylist = []
for i in range(1, 11):
    mylist.append((i*2))
    print(i)

mylist = [0] * 10
for i in range(10):
    mylist[i] = 2*(i+1)

for x in numbers:
    for y in range(4):
        print(x)

# 0+0 / 0+1 / 1+0 / 1+1 / 2+0 / 2+1
for i in range(3):
    for j in range(2):
        print(i+j)

# 0,4 / 0,5 / 0,6 / 1,4 / 1,5 / 1,6
for i in [0,1]:
    for j in [4, 5, 6]:
        print(i,j)

# 이게 다 같음
numbers = [5, 2, 3]
for i in numbers:
    for j in range(3):
        print(i,j)
for i in numbers:
    for j in range(3):
        print(i)
[i for i in numbers for j in range(3)]
repeated_list = [x for x in numbers for _ in range(3)]

# [x == "banana" for x in fruits]
fruits = ["apple", "apple", "banana", "cherry"]
mylist=[]
for x in fruits:
    mylist.append(x=="banana")

# 바나나의 위치
fruits = np.array(fruits)
int(np.where(fruits == "banana")[0][0])

# 원소 순서를 반대로 변경
fruits.reverse()
fruits.append("pineapple")

fruits.insert(2, "test") # 그 자리에 끼워넣기

# 원소 제거
fruits.remove("test")

# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])
# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])
# 논리형 벡터 생성
mask = ~np.isin(fruits, items_to_remove) # banana, apple 아닌 게 True
mask = ~np.isin(fruits, ["banana", "apple"])
# 논리형 벡터를 사용하여 항목 제거
filtered_fruits = fruits[mask]

# 인덱스 공유하기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 10
# in 뒤에 mylist_b를 넣었을 때 안 되었던 이유는 처음 원소가 2라서 3번째 자리를 수정해서
for i in range(10):
    mylist[i] = mylist_b[i]

# 퀴즈: 홀수번째 위치에 있는 숫자들만 mylist에 가져오기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 5

# 1번
for i in range(5):
    mylist[i] = mylist_b[::2][i]

# 2번
for i in range(5):
    mylist[i] = mylist_b[2*i]

# 리스트 Comprehension으로 바꾸는 방법
# 바깥은 무조건 대괄호로 묶어줌: 리스트로 반환하기 위해서
# for 루프의 : 는 생략
# 실행 부분을 먼저 써줌
# 결과값을 발생하는 표현만 남겨둠
mylist = []
[(i*2) for i in range(1, 11)]
[x for x in numbers]
