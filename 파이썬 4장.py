import pandas as pd
df = pd.DataFrame({'name': ['김지훈', '이유진', '박동현', '김민지'],
'english': [90, 80, 60, 70], 'math': [50, 60, 100, 20]})
df
type(df) # pandas.core.frame.DataFrame
df['english'] # 0    90
              # 1    80
              # 2    60
              # 3    70
              # Name: english, dtype: int64
df[['name', 'english']]
type(df['english']) # pandas.core.series.Series
sum(df['english'])/4
sum(df['math'])/4

!pip install openpyxl
df_exam = pd.read_excel('Data/excel_exam.xlsx')
df_exam = pd.read_excel('C:/Doit_Python-main/Data/excel_exam.xlsx')
df_exam

df_exam['math']
df_exam['english']
df_exam['science']

sum(df_exam['english'])/20
sum(df_exam['science'])/20
len(df_exam)
df_exam.shape # (20, 5)
df_exam.size
sum(df_exam['english'])/len(df_exam)
sum(df_exam['science'])/len(df_exam)

x = [1, 2, 3, 4, 5]
x
len(x)

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df
len(df)

df_exam_novar = pd.read_excel('C:/Doit_Python-main/Data/excel_exam_novar.xlsx')
df_exam_novar
df_exam_novar = pd.read_excel('C:/Doit_Python-main/Data/excel_exam_novar.xlsx', header = None)
df_exam_novar

df_exam = pd.read_excel('Data/excel_exam.xlsx', sheet_name = 'Sheet2')
df_exam

# 없는 칼럼 생성
df_exam['total'] = df_exam['math'] + df_exam['english'] + df_exam['science']
df_exam

df_exam['mean'] = df_exam['total']/3
df_exam

import numpy as np
df_exam[(df_exam["math"] > 50) & (df_exam["english"] > 50)]
df_exam[(df_exam["math"] > sum(df_exam["math"])/20) & (df_exam["english"] < sum(df_exam["english"])/20)]
mean_m = np.mean(df_exam["math"])
mean_e = np.mean(df_exam["english"])
df_exam[(df_exam["math"] > mean_m) & (df_exam["english"] < mean_e)]
df_exam[df_exam["nclass"] == 3]
df_exam[df_exam["nclass"] == 3][["math", "english", "science"]]
df_nc3 = df_exam[df_exam["nclass"] == 3]
df_nc3[["math", "english", "science"]]
df_nc3[1:2]
df_exam
df_exam[:10]
df_exam[7:16]
df_exam[0:10:2]
df_exam.sort_values("math", ascending = False)
df_exam.sort_values(["nclass", "math"], ascending=[True, False])

a = np.array([4, 2, 5, 3, 6])
a[2]
np.where(a>3) # 튜플
np.where(a>3, "Up", "Down") # numpy.array
df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down")
df_exam

df_csv_exam = pd.read_csv('C:/Doit_Python-main/Data/exam.csv')
df_csv_exam

df_midterm = pd.DataFrame({'english': [90, 80, 60, 70], 'math': [50, 60, 100, 20],
'nclass': [1, 1, 2, 2],})
df_midterm
df_midterm.to_csv('C:/Doit_Python-main/Data/output_newdata.csv')
df_midterm.to_csv('C:/Doit_Python-main/Data/output_newdata.csv', index = False)
