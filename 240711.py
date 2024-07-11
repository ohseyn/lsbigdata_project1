# 데이터 타입
x = 15.34
print(x, "는", type(x), "형식입니다.", sep=' ')
#sep는 입력값 사이에 넣을 값값

# 문자형 데이터 예제
a= "Hello, world!"
b = 'python programming'

print(a, type(a))
print(b, type(b))

# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""
print(ml_str, type(ml_str))

# 리스트 생성 예제
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "Hello", [1, 2, 3]]
print("Fruits:", fruits)
print("Numbers:", numbers)
print("Mixed List:", mixed_list)

# 튜플 생성 예제
a_tp= (10, 20, 30, 40, 50)
a_ls = [10, 20, 30, 40, 50]
a_tp[1] = 25
a_tp
a_ls[1] = 25
a_ls
type(a_ls)

b_int= (42)
b_tp= (42,)

# 인덱싱
print("첫번째 좌표:", a1[0])

# 슬라이싱
print("좌표:", a_tp[3:]) # 해당 인덱스 이상
print("좌표:", a_tp[:3]) # 해당 인덱스 미만
print("좌표:", a_tp[1:3]) # 해당 인덱스 이상 & 미만
print("좌표:", a_ls[3:]) # 해당 인덱스 이상
print("좌표:", a_ls[:3]) # 해당 인덱스 미만
print("좌표:", a_ls[1:3]) # 해당 인덱스 이상 & 미만

# 사용자 정의 함수
def min_max(numbers):
 return min(numbers), max(numbers)

a = [1, 2, 3, 4, 5]
type(a)
result = min_max(a)
result
type(result)
print("Minimum and maximum:", result)

# 딕셔너리
person = {
  'name': 'John',
  'age': 30,
  'city': ('New York', 'LA')
}
seoyeon = {
  'name': '오서연',
  'age': 23,
  'city': 'Seoul'
}
print("Person: ", person)
print("Person: ", seoyeon)

person_city = person.get('city')
person_city

# 집합
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits)
type(fruits)

# 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)

empty_set.add('apple')
empty_set
empty_set.add('banana')
empty_set.add('apple')
empty_set.remove('banana')
empty_set
empty_set.discard('banana')
empty_set
empty_set.remove('banana')

# 집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits) # 합집합
intersection_fruits = fruits.intersection(other_fruits) #교집합
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)

p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) 

age = 10
is_active = True
is_greater = 10 > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

# 조건문
a = 3
if( a == 2):
  print("a는 2와 같습니다.")
else:
  print("a는 2와 같지 않습니다.")

# 형 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

set_example = {'a', 'b', 'c'} # 집합(중괄호)
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)
