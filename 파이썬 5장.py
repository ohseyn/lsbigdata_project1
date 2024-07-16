import pandas as pd
import numpy as np

# 데이터 탐색 함수
# head()
# tail()
# shape
# info()
# describe()

exam = pd.read_csv("data/exam.csv")
exam.head(10) # 기본은 다섯 번째 행까지 출력, 괄호에 숫자를 입력하면 입력한 행까지 출력
exam.tail(10) # 기본은 뒤에서 다섯 번째 행까지 출력, 괄호에 숫자를 입력하면 뒤에서부터 입력한 행까지 출력
exam.shape # 튜플 # 몇 행, 몇 열열
exam.shape() # 에러
exam.info() # 변수 속성
exam.describe() # 요약 통계량 

# 메서드(함수) vs 속성(어트리뷰트)
# 속성은 괄호가 없지만 메서드는 괄호가 있음
# 내장 함수 / 패키지 함수(pd) / 메서드(로드, 객체(변수) 생성, df)

type(exam) # pandas DataFrame
var = [1, 2, 3]
type(var) # list
exam.head()
var.head() # 오류(head 메서드가 없으므로)

exam2 = exam.copy()
exam2.rename(columns = {"nclass" : "class"}) # 임시 수정(exam2는 수정 안 되어 있음)
exam2 = exam2.rename(columns = {"nclass" : "class"}) # 진짜 수정(업데이트를 해줘야 함)
exam2["total"] = exam2["math"] + exam2["english"] + exam2["science"]

# np.where()는 지정한 조건에 맞지 않을 때 서로 다른 값을 반환하는 기능
# np.where(조건, 조건에 맞을 때 부여, 조건에 맞지 않을 때 부여)
exam2["test"] = np.where(exam2["total"] >= 200, "pass", "fail")
exam2["test2"] = np.where(exam2["total"] >= 200, "A", 
                 np.where(exam2["total"] >= 100, "B", "C"))
exam2["test2"].value_counts()
exam2.head()

# snippet Shift + Tab
import matplotlib.pyplot as plt
count_test = exam2["test"].value_counts()
count_test.plot.bar()
count_test.plot.bar(rot = 0)
plt.show()
plt.clf()

df = pd.DataFrame({"lab": ["A", "B", "C"], "val": [10, 30, 20]})
ax = df.plot.bar(x="lab", y="val", rot = 0)
plt.show()
plt.clf()

exam2["test2"].isin(["A", "C"])
