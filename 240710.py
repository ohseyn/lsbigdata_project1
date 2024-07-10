# ctrl + Enter
# shift + 화살표: 블록
# ctrl + A: 전체 선택

a=1
a
a=10
a
a = "안녕하세요!"
a
a = '안녕하세요!'
a
a = "'안녕하세요!' 라고 말했다."
a
a = '"안녕하세요!" 라고 말했다.'
a
a= [1, 2, 3] #리스트
a
b= [4, 5, 6]
b
a+b

a = '안녕하세요!'
a
b = 'LS 빅데이터 스쿨!'
b

a+b
a + ' ' + b

print(a)

a = 10
b = 3.3
print("a + b = ", a + b)
print("a - b = ", a - b)
print("a * b = ", a * b)
print("a / b = ", a / b)
print("a % b = ", a % b)
print("a // b = ", a // b)
print("a ** b = ", a ** b)

a='안녕하세요!'
b='4, 5, 6'
a+b

# Shift + Alt + 아래 화살표: 아래로 복사
# Ctrl + Alt + 아래 화살표: 커서 여러 개

a=10

(a ** 3) // 7
(a ** 3) % 7

a == b
a != b
b
b = 3.3
a < b
a >= b

# 2에 4승과 12453을 7로 나눈 몫을 더해서 8로 나눴을 때 나머지
# 9의 7승을 12로 나누고, 36452를 253으로 나눈 나머지에 곱한 수 
# 중에 큰 것은?

a = ((2 ** 4) + (12453 // 7)) % 8
b = ((9 ** 7) / 12) * (36452 % 253)

a < b

user_age = 14
is_adult = user_age > 18
print("성인입니까?", is_adult)

False = 3 # 논리형 불리언(Boolean)의 값

TRUE = True
a = "True" # 문자열
b = TRUE # TRUE가 정의되어 있지 않음 -> 변수로 인식
c = true # true가 정의되어 있지 않음 -> 변수로 인식
d = True # 논리 연산자 값의 True

a = True
b = False
a and b
a or b
not a

# True, False
# and(곱셈): 둘 다 참일 때만 True
# or(더하기): 둘 다 거짓일 때만 False
# True: 1 / False: 0

# or
True + False
True + True
False + False
False + True

# and
True * False
True * True
False * False
False * True

a = False
b = False
a or b
min(a + b, 1) #최솟값

# 한 번도 안 산 사람 / 한 번이라도 산 사람
# or = 0 / or >= 1

# 복합 대입 연산자
a = 3
a += 10
a -= 4
a %= 3
a += 12
a **= 2
a /= 7
a

strl = "Hello! "
repeated_str = strl * 3
print("Repeated string: ", repeated_str)

strl * -2

# 정수: integer
# 실수: float / double

# 단항 연산자
x = 5
print("Original x:", x)
print("+x: ", +x)
print("-x: ", -x)
print("~x: ", ~x)

bin(5) #'0b101': 0b는 이진수, 101(2): 5
bin(-5) #'-0b101' 
bin(~5) #'-0b110' -> -6(110)이 됨. 반전이 1의 보수이므로(2의 보수가 아님) 1을 더하지 않아야 함. 비트 반전만 이루어짐.
bin(~-5) # 0b100'

max(3, 4) #최댓값
var1 = [1, 2, 3]
sum(var1) #총합

import pydataset
pydataset.data()

df = pydataset.data("AirPassengers")
df

import pandas as pd
import numpy as np

#time에서 해마다 12개의 데이터가 있어서 월별 탑승자를 나타냄

# . 현재 폴더
# .. 상위 폴더

# show folder in new window: 해당 위치 탐색기
# CMD- dir: 디렉토리

# PowerShell
# ls: 파일 목록 
# cd: 폴더 이동
# cd ..: 상위 폴더
# cd ..\.. 두 개의 상위 폴더 이동 
# cd .\폴더명\폴더명 하위 폴더 이동
# Tab/Shift Tab 자동완성
# cls 화면 정리
