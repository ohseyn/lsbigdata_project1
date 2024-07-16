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
