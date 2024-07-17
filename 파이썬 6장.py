import pandas as pd
import numpy as np

# 데이터 전처리 함수
# query(): 조건에 맞는 행을 걸러냄
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()

exam = pd.read_csv("data/exam.csv")
exam.query("nclass == 1") # exam[exam["nclass"] == 1]
exam.query("nclass != 1")
exam.query("math > 50")
exam.query("math < 50")
exam.query("english >= 50")
exam.query("english <= 80")
exam.query("nclass == 1 & math >= 50")
exam.query("nclass == 2 and english >= 80")
exam.query("math >= 90 | english >= 90")
exam.query("english < 90 | science < 50")
exam.query("nclass == 1 or nclass == 3 or nclass == 5")
exam.query("nclass in [1, 3, 5]")
exam.query("nclass not in [1, 2]") # exam[~exam["nclass"].isin([1,2])]

exam["nclass"] # 시리즈
exam[["nclass"]] # 데이터 프레임
exam[["id", "nclass"]] # 여러 변수
exam.drop(columns = "math") # 미반영. 업데이트 안 했기 때문(exam = exam.drop(columns = "math"))
exam.drop(columns = ["math", "english"]) # 여러 변수 제거
exam.query("nclass == 1")[["math", "english"]] # query()와 [] 조합
exam.query("nclass == 1")\
            [["math", "english"]]\
            .head() # 일부만 출력 

# 정렬하기
exam.sort_values(["nclass", "english"], ascending = [True, False])
exam = exam.assign(total = exam["math"] + exam["english"] + exam["science"],
            mean = (exam["math"] + exam["english"] + exam["science"])/3)\
            .sort_values("total", ascending = False)

exam2 = pd.read_csv("data/exam.csv")
exam2 = exam2.assign(total = lambda x: x["math"] + x["english"] + x["science"],
                     mean = lambda x: x["total"]/3).sort_values("total", ascending = False)

# 그룹을 나눠 요약을 하는 .groupby() + .agg() 콤보                     
exam2.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass").agg(mean_math = ("math", "mean"), mean_english = ("english", "mean"), 
        mean_science = ("science", "mean"))

import pydataset
mpg = pd.read_csv("data/mpg.csv")
mpg
mpg.info()
mpg.query('category == "suv"')\
        .assign(total = (mpg["hwy"] + mpg["cty"])/2)\
        .groupby("manufacturer")\
        .agg(mean_tot = ("total", "mean"))\
        .sort_values("mean_tot", ascending = False)\
        .head()

test1 = pd.DataFrame({"id" : [1, 2, 3, 4, 5], "midterm" :[60, 80, 70, 90, 85]})
test2 = pd.DataFrame({"id" : [1, 2, 3, 40, 5], "final" :[70, 83, 65, 95, 80]})

# Left Join
total = pd.merge(test1, test2, how = "left", on = "id") # 4번 학생 final: NaN

# Right Join
total = pd.merge(test1, test2, how = "right", on = "id") # 40번 학생 midterm: NaN

# Inner Join
total = pd.merge(test1, test2, how = "inner", on = "id") # 공통적으로 있는 애들만 남김(필터도 있는 느낌)

# Outer Join
total = pd.merge(test1, test2, how = "outer", on = "id") # 합집합(중복 먼저, 그 뒤에 중복 아닌 애들 추가)

name = pd.DataFrame({"nclass" : [1, 2, 3, 4, 5], "teacher" :["kim", "lee", "park", "choi", "jung"]})

exam_new = pd.merge(exam, name, how = "left", on = "nclass")

# 데이터를 세로로 쌓는 법(데이터 변수명이 같아야 함. 다르면 pd.rename()으로 똑같이 맞춰야 함)

score1 = pd.DataFrame({"id" : [1, 2, 3, 4, 5], "score" :[60, 80, 70, 90, 85]})
score2= pd.DataFrame({"id" : [6, 7, 8, 9, 10], "score" :[70, 83, 65, 95, 80]})

score_all = pd.concat([score1, score2])

pd.concat([test1, test2], axis=1)
