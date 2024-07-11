import pandas as pd
data = pd.read_excel('C:/Doit_Python-main/Data/excel_exam.xlsx')
print(data)

import seaborn as sns
import matplotlib.pyplot as plt
var = ['a', 'a', 'b', 'c']
var
seaborn.countplot(x = var, color = 'red')
seaborn.countplot(x = var, hue = var)
plt.show()
plt.clf()

df = sns.load_dataset('titanic')
df
sns.countplot(data = df, x = 'sex', hue = 'sex')
sns.countplot(data = df, x = 'class', hue = 'alive')
sns.countplot(data = df, y = 'class', hue = 'alive')
plt.show()
plt.clf()

import sklearn.metrics
sklearn.metrics.accuracy_score()

from sklearn import metrics
metrics.accuracy_score()

from sklearn.metrics import accuracy_score
accuracy_score()

import pydataset
pydataset.data()

df = pydataset.data('mtcars')
df
