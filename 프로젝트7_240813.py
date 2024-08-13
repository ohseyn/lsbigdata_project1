import pandas as pd
import numpy as np

df = pd.read_csv("Data/Ames population data.csv")
df = df.iloc[5::, :2]
df_age = df.iloc[0:13, :2].reset_index(drop=True)
df_race = pd.concat([df.iloc[72:76, :2], df.iloc[77:84, :2]]).reset_index(drop=True)
