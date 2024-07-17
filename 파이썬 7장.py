import pandas as pd
import numpy as np

df = pd.DataFrame({"sex" :["M", "F", np.nan, "M", "F"],
                   "score" : [5, 4, 3, 4, np.nan]})
df["score"] +1
pd.isna(df).sum()

df.dropna(subset = "score") # score 변수에서 결측치 제거
df.dropna(subset = ["score", "sex"]) # 여러 변수 결측치 제거
df.dropna() # 모든 변수 결측치 제거

exam = pd.read_csv("data/exam.csv")

# 데이터 프레임 location을 사용한 인덱싱
# exam.loc[행 인덱스, 열 인덱스]
exam.loc[[0],]
exam.iloc[0:2, 0:4]
exam.loc[[2, 7, 14], ["math"]] = np.nan
exam.iloc[[2, 7, 14], 2] = 3

df.loc[df["score"] == 3.0, ["score"]] = 4

# 수학 점수 50점 이하인 학생들 점수 50점으로 상향 조정!
exam.loc[exam["math"] <= 50, ["math"]] = 50
# 영어 점수 90점 이상 90으로 하향 조정
# iloc 숫자 벡터면 잘 돌아감

# iloc 조회는 안됨
exam.loc[exam["english"] >= 90, "english"]

# iloc을 사용해서 조회하려면 무조건 숫자 벡터가 들어가야 함.
exam.iloc[exam["english"] >= 90, 3] = 90 # 실행 안됨
exam.iloc[exam[exam["english"] >= 90].index, 3] # 실행 됨
# 튜플 안에 numpy.array를 꺼내오는 거임. # np.where도 튜플이라 [0] 사용해서 꺼내오면 됨.
exam.iloc[np.where(exam["english"] >= 90)[0], 3]
exam.iloc[np.array(exam["english"] >= 90), 3] # index 벡터도 작동

# math 점수 50 이하 "-" 변경
exam.loc[exam["math"] <= 50, ["math"]] = "-"

# "-" 결측치를 수학점수 평균으로 바꾸기
# 1번
math_mean = exam.loc[(exam["math"] != "-"), "math"].mean()
exam.loc[exam["math"] == "-", "math"] = math_mean

# 2번
math_mean = exam.query('math not in ["-"]')["math"].mean()
exam.loc[exam["math"] == "-", "math"] = math_mean

# 3번
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam.loc[exam["math"] == "-", ["math"]] = math_mean

# 4번
exam.loc[exam["math"] == "-", ["math"]] = np.nan
math_mean = exam["math"].mean()
exam.loc[pd.isna(exam["math"]), ["math"]] = math_mean

# 5번
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam["math"] = exam["math"].replace("-", math_mean)

# 6번
math_mean = np.nanmean(np.array([np.nan if x == "-" else float(x) for x in exam["math"]]))
vector = np.array([float(x) if x != "-" else np.nan for x in exam["math"]])
exam["math"] = np.where(exam["math"] == "-", math_mean, exam["math"])
