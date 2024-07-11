import pandas as pd
data = pd.read_excel('C:/Doit_Python-main/Data/excel_exam.xlsx')
print(data)

import seaborn as sns
import matplotlib.pyplot as plt
var = ['a', 'a', 'b', 'c']
var
seaborn.countplot(x = var)
plt.show()

df = sns.load_dataset('titanic')
df
sns.countplot(data = df, x = 'class', hue = 'alive')
plt.show()
plt.clf()
