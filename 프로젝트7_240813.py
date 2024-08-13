import pandas as pd
import numpy as np

# 데이터 불러오기, 데이터 전처리
df = pd.read_csv("Data/Ames population data.csv")
df = df.iloc[5::, :2]
df_age = df.iloc[0:13, :2].reset_index(drop=True)
df_age['Ames city, Iowa!!Estimate'] = df_age['Ames city, Iowa!!Estimate'].str.replace(',', '').astype(int)
df_race = pd.concat([df.iloc[72:76, :2], df.iloc[77:84, :2]]).reset_index(drop=True)
df_race['Ames city, Iowa!!Estimate'] = df_race['Ames city, Iowa!!Estimate'].str.replace(',', '').astype(int)

# 나이
import plotly.graph_objects as go
labels = df_age["Label (Grouping)"]
values = df_age["Ames city, Iowa!!Estimate"]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

fig.update_layout(
    title_text="Ames 나이 분포",
    title_font_size=24,
    title_x=0.5,
    annotations=[dict(text='나이', x=0.5, y=0.5, font_size=20, showarrow=False)],
    showlegend=True
)

fig.show()

# 인종
labels = df_race["Label (Grouping)"]
values = df_race["Ames city, Iowa!!Estimate"]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

fig.update_layout(
    title_text="Ames 인종 분포",
    title_font_size=24,
    title_x=0.5,
    title_y=0.9,
    annotations=[dict(text='인종', x=0.5, y=0.5, font_size=20, showarrow=False)],
    showlegend=False
)

fig.show()