import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from palmerpenguins import load_penguins
from itables import show

penguins = load_penguins()
# show(penguins, buttons = ["copy", "excel", "pdf"])

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species"
    # trendline = "ols"
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

#================================================

from plotly.subplots import make_subplots

# 여러 개의 서브플롯을 가진 객체 생성
fig_subplot = make_subplots(
    rows=1, cols=3, # 1행 3열
    subplot_titles=('Adelie', 'Gentoo', 'Chinstrap')
)

fig_subplot.add_trace(
  {
   'type' : 'scatter',
   'mode' : 'markers',
   'x' : penguins.query('species=="Adelie"')['bill_length_mm'],
   'y' : penguins.query('species=="Adelie"')['bill_depth_mm'],
   'name' : 'Adelie'
  },
  row=1, col=1 # 첫 번째 행 첫 번째 열
 )

fig_subplot.add_trace(
  {
   'type' : 'scatter',
   'mode' : 'markers',
   'x' : penguins.query('species=="Gentoo"')['bill_length_mm'],
   'y' : penguins.query('species=="Gentoo"')['bill_depth_mm'],
   'name' : 'Gentoo'
  },
  row=1, col=2 # 첫 번째 행 두 번째 열
 )

fig_subplot.add_trace(
  {
   'type' : 'scatter',
   'mode' : 'markers',
   'x' : penguins.query('species=="Chinstrap"')['bill_length_mm'],
   'y' : penguins.query('species=="Chinstrap"')['bill_depth_mm'],
   'name' : 'Chinstrap'
  },
  row=1, col=3 # 첫 번째 행 세 번째 열
 )

fig_subplot

fig_subplot.update_layout(
    title=dict(text="펭귄종별 부리 길이와 부리 깊이", x=0.5)
)

#========================

species_list = ["Adelie", "Gentoo", "Chinstrap"]
# enumerate: 순서가 있는 자료형을 입력으로 받았을 때, 인덱스와 값을 포함하여 리턴
# for문과 함께 자주 사용
# 인덱스와 값을 동시에 접근하면서 루프를 돌리고 싶을 때 사용
for i, species in enumerate(species_list, start=1):
    print(i)
    print(species)

fig = make_subplots(
    rows=1, 
    cols=3, 
    subplot_titles=["Adelie", "Gentoo", "Chinstrap"], 
    horizontal_spacing=0.05, # 서브플롯 사이의 간격을 조정
    shared_xaxes=True) # 모든 서브플롯이 동일한 x축을 공유

# x축의 범위를 설정
min_bill_length = penguins['bill_length_mm'].min()
max_bill_length = penguins['bill_length_mm'].max()

# 데이터프레임의 각 종별로 반복
# 각 종의 데이터를 필터링하여 subset 변수에 저장
# i는 현재 종의 열 위치, species는 종의 이름
for i, species in enumerate(penguins['species'].unique(), 1):
    # 현재 종 (species)에 해당하는 데이터만 필터링하여 subset이라는 새로운 데이터프레임을 생성
    # 예를 들어, Adelie면 subset에는 Adelie의 모든 데이터가 포함
    subset = penguins[penguins['species'] == species]
    fig.add_trace(
        go.Scatter(
            x=subset['bill_length_mm'], 
            y=subset['bill_depth_mm'], 
            mode='markers',
            marker=dict(size=7, line=dict(width=1)),
            name=f'{species}' # 레이블을 현재 종의 이름으로 설정
        ),
        row=1, col=i # 서브플롯 (row=1, col=i)에 추가
        # i: 현재 종에 대한 서브 플롯의 열 위치를 지정
    )

# x축과 y축의 레이블 설정
fig.update_xaxes(title_text="부리 길이 (mm)", range=[min_bill_length, max_bill_length], tickmode='auto')
fig.update_yaxes(title_text="부리 깊이 (mm)", tickmode='auto') # 축의 눈금 (tick) 모드를 자동으로 설정

fig.update_layout(height=400, width=1000, title_text="펭귄 종별 부리 치수", title_x=0.5)
fig.update_layout(showlegend=False) # 범례를 숨김

# 축의 눈금을 표시
fig.update_xaxes(showticklabels=True)
fig.update_yaxes(showticklabels=True)

fig

#=============================================

fig = make_subplots(
    # 3행 3열의 서브플롯 그리드 생성
    rows=3, cols=3,
    # specs: 서브플롯의 레이아웃을 정의
    # 첫 번째 행은 전체 데이터를 위한 하나의 큰 서브플롯을 포함 
    # 두 번째 행은 비어 있음
    # 세 번째 행은 3개의 작은 서브플롯이 있음
    specs=[[{'colspan': 3, 'rowspan': 2}, None, None],
           [None, None, None],
           [{'colspan': 1}, {'colspan': 1}, {'colspan': 1}]],
    subplot_titles=["전체 데이터", "Adelie", "Gentoo", "Chinstrap"],
    row_heights=[0.4, 0.4, 0.2],  # 각 행의 높이를 비율로 설정
    shared_xaxes=True, # 모든 서브플롯에서 x축을 공유
    horizontal_spacing=0.05, # 서브플롯 간의 간격을 설정
    vertical_spacing=0.1)

# items(): 딕셔너리의 (키, 값) 을 튜플로 반환
colors = {
    "Adelie": "blue",
    "Gentoo": "red",
    "Chinstrap": "green"
}

# 전체 데이터를 하나의 서브플롯에 추가
# 딕셔너리의 각 항목을 반복
# 전체 데이터니 col=i를 해줄 필요도 없고, for 문에 i가 들어갈 필요도 없음
for species, color in colors.items():
    # 현재 종 (species)에 해당하는 데이터만 필터링하여 subset이라는 새로운 데이터프레임을 생성
    # 예를 들어, Adelie면 subset에는 Adelie의 모든 데이터가 포함
    subset = penguins[penguins['species'] == species]
    # 생성된 트레이스를 첫 번째 행의 첫 번째 열(전체 데이터 서브플롯)에 추가
    fig.add_trace(
        go.Scatter(
            x=subset['bill_length_mm'], 
            y=subset['bill_depth_mm'], 
            mode='markers',
            marker=dict(size=8, color=color),
            name=f'{species}' # 레이블을 현재 종의 이름으로 설정
        ),
        row=1, col=1 # 첫 번째 행의 첫 번째 열(전체 데이터 서브플롯)에 추가
    )

# 각 종별로 데이터를 개별 서브플롯에 배치
# 각 종에 대해 인덱스와 종 이름을 함께 얻음. 인덱스는 1부터 시작
for i, species in enumerate(penguins['species'].unique(), 1):
    # 현재 종 (species)에 해당하는 데이터만 필터링하여 subset이라는 새로운 데이터프레임을 생성
    # 예를 들어, Adelie면 subset에는 Adelie의 모든 데이터가 포함
    subset = penguins[penguins['species'] == species]
    # 각 종별 데이터를 세 번째 행의 각 열에 추가
    fig.add_trace(
        go.Scatter(
            x=subset['bill_length_mm'], 
            y=subset['bill_depth_mm'], 
            mode='markers',
            marker=dict(size=7, line=dict(width=1), color=colors[species]),
            name=f'{species}' # 레이블을 현재 종의 이름으로 설정
        ),
        row=3, col=i # 세 번째 행의 i번째 열에 추가
        # i: 현재 종에 대한 서브 플롯의 열 위치를 지정
    )

# x축과 y축에 제목 추가
fig.update_xaxes(title_text="부리 길이 (mm)")
fig.update_yaxes(title_text="부리 깊이 (mm)")

# 전체 레이아웃 설정(높이, 너비, 제목)
fig.update_layout(height=900, width=1000, title_text="펭귄 종별 부리 치수", title_x=0.5)
fig