import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species"
)

fig.show()

fig.update_layout(
    title={"text": "팔머펭귄",
    "x": 0.5, "y": 0.5}
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

