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
type(df['english']) # pandas.core.series.Series
sum(df['english'])/4
sum(df['math'])/4

df_fruit = pd.DataFrame({'제품': ['사과', '딸기', '수박'], 
                         '가격': [1800, 1500, 3000], '판매량': [24, 38, 13]})
df_fruit
sum(df_fruit['가격'])/len(df_fruit)
sum(df_fruit['판매량'])/len(df_fruit)

df_exam = pd.read_excel('C:/Doit_Python-main/Data/excel_exam.xlsx')
df_exam
sum(df_exam['english'])/20
sum(df_exam['science'])/20
len(df_exam)
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

df_csv_exam = pd.read_csv('C:/Doit_Python-main/Data/exam.csv')
df_csv_exam

df_midterm = pd.DataFrame({'english': [90, 80, 60, 70], 'math': [50, 60, 100, 20],
'nclass': [1, 1, 2, 2],})
df_midterm
df_midterm.to_csv('C:/Doit_Python-main/Data/output_newdata.csv')
df_midterm.to_csv('C:/Doit_Python-main/Data/output_newdata.csv', index = False)
