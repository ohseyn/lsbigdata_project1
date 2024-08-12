import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from palmerpenguins import load_penguins
from itables import show

penguins = load_penguins()
show(penguins, buttons = ["copy", "excel", "pdf"])

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species"
)

fig.show()

# xref: 타이틀의 수평 위치를 측정할 기준을 설정
# x: 타이틀을 플롯의 너비에 대한 비율로 설정
# y: 타이틀을 플롯의 높이에 대한 비율로 설정
fig.update_layout(
    title={"text": "팔머펭귄",
    "x": 0.5, "y": 0.5} # xref에서 x축, y축 위치 설정
)

# xanchor: 타이틀의 수평 위치를 결정
fig.update_layout(
    title={"text": "팔머펭귄",
    "x": 0.5, "xanchor": "center"} # 타이틀의 중앙이 x 좌표에 맞춰짐
)

# p.70: span 1개 안에 span 3개가 있음.
# <span>
#     <span style='font-weight:bold'> ... </span>
#     <span> ... </span>
#     <span> ... </span>
# </span>

fig.update_layout(
    title={"text": "<span style='color:blue;font-weight:bold;'>팔머펭귄</span>",
    "x": 0.5}
)

