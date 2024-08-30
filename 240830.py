# 함수
def g(x=3):
    result = x + 1
    return result
g()

# 함수 내용 확인
import inspect
# 문자열 출력
print(inspect.getsource(g))

# if문
x=3
if x>4:
    y=1
else:
    y=2

# if문 한 줄 버전
x=3
y = 1 if x >4 else 2

# list comprehension
x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0 else "음수" for value in x]

# 조건 3가지 if문
x = 0
if x > 0:
    result = "양수"
elif x == 0:
    result = "0"
else:
    result = "음수"

# 조건 3가지 numpy
import numpy as np
x = np.array([1, -2, 3, -4, 0])
conditions = [x > 0, x == 0, x < 0]
choices = ["양수", "0", "음수"]
# select: 여러 개의 조건을 처리
# 각 조건에 대한 결과를 리스트로 작성하여 
# 조건에 따라 결과를 반환
result = np.select(conditions, choices, x)

# for loop
for i in range(1, 4):
     print(f"Here is {i}")

# for loop 한 줄 버전
[f"Here is {i}" for i in range(1, 4)]

name = "Ohseyn"
age = 25
f"Hello, my name is {name} and I am {age} years old."

# 각각 대응하는 내용을 출력
names = ["John", "Alice"]
ages = np.array([25, 30])
greetings = [f"이름: {name}, 나이: {age}"
             for name, ages in zip(names, ages)]

# zip
names = ["John", "Alice"]
ages = np.array([25, 30])
zipped = zip(names, ages)

for name, age in zipped:
    print(f"이름: {name}, 나이: {age}")

# while
i = 0
# 참일 때 돌아감
while i <= 10:
    i += 3
    print(i)

# while, break
i = 0
while True:
    i += 3
    if i > 10:
        break
    print(i)

# DataFrame
import pandas as pd
data = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6]
})

# max
data.apply(max, axis=0)
data.apply(max, axis=1)

# sum
data.apply(sum, axis=0)
data.apply(sum, axis=1)

# apply
def my_func(x, const=3):
    return max(x)**2 + const

my_func([3, 4, 10], 5)

data.apply(my_func, axis=1)
data.apply(my_func, axis=0)
data.apply(my_func, axis=0, const=5)

# 세로로 채워주게 함
array_2d = np.arange(1, 13).reshape((3,4), order="F")
np.apply_along_axis(max, axis=0, arr=array_2d)

# 변수 저장하는 공간이 나뉘어짐
# 전체: y=2 ⊃ my_func: y=1
y = 2
def my_func(x):
     y = 1
     result = x + y
     return result
my_func(3)
y

# global y
y = 2
def my_func(x):
     global y
     y = y + 1
     result = x + y
     return result
my_func(3)
y

# global y, function
y = 2
def my_func(x):
     global y
     def my_f(k):
         return k**2
     y = my_f(x) + 1
     result = x + y
     return result
# my_f(3)
my_func(3)
y

# 입력값이 몇 개일지 모를 땐 *
# 입력값을 리스트로 받아와서 돌림
def add_many(*args):
    result = 0
    for i in args:
        result += i
    return result

# for 문 안에서 return이 끝났기 때문에 1 넣고 끝남
# def add_many(*args):
#     result = 0
#     for i in args:
#         result += i
#         return result

add_many(1, 2, 3)

def first_many(*args):
    return args[0]

first_many(1, 2, 3)

def add_mul(choice, *args):
    if choice == "add":
        result = 0
        for i in args:
            result += i
    elif choice == "mul":
        result = 1
        for i in args:
            result *= i
    return result

add_mul("mul", 5, 4, 3, 1)

# 별표 두 개(**)는 입력값을 딕셔너리로 만들어줌
dict_a={"age": 25, "name": "Ohseyn"}
dict_a["age"]
dict_a["name"]

def my_twostars(choice, **kwargs):
    if choice == "first":
        return print(kwargs["age"])
    elif choice == "second":
        return print(kwargs["name"])
    else:
        return print(kwargs)
    
my_twostars("first", age=25, name="Ohseyn")
my_twostars("second", age=25, name="Ohseyn")
my_twostars("all", age=25, name="Ohseyn")
