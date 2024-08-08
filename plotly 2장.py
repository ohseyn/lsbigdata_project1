!pip install plotly
import plotly.graph_objects as go
import plotly.express as px

fig = go.Figure()
fig.show()

df_covid19_100 = pd.read_csv("Data/df_covid19_100.csv")
df_covid19_100.info()

# data: 데이터를 표현하는 트레이스를 구성하는 세부 속성들
# 트레이스: Plotly로 시각화 할 수 있는 그래픽적 데이터 표현 방법
fig = go.Figure(
    data = [{
        "type": "scatter", 
        "mode": "markers", # 점
        "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"], # 일자
        "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"], # 건수
        "marker": {"color": "red"}
        },
        {
        "type": "scatter", 
        "mode": "lines", # 선
        "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
        "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
        "line": {"color": "blue"}
        }]
)
fig.show()

# 여백
# layout: 데이터를 표현하는 트레이스와 관련되지 않는 시각화의 나머지 속성들을 정의
margins_P = {"t": 50, "b": 25, "l": 25, "r": 25} # top, bottom, left, right
fig = go.Figure(
    data = [{
        "type": "scatter", 
        "mode": "markers", 
        "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
        "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
        "marker": {"color": "red"}
        },
        {
        "type": "scatter", 
        "mode": "lines", 
        "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
        "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
        "line": {"color": "blue"}
        }],
    layout = {
        "title": "코로나19 발생 현황", # 전체 제목
        "xaxis": {"title": "날짜", "showgrid": False}, # x축 layout 속성 설정
        "yaxis": {"title": "확진자수"}, # y축 layout 속성 설정
        "margin": margins_P # 여백 설정
    }
)
fig.show()

#============================================================
# 프레임 속성
# 애니메이션 프레임 생성
frames = [] # 리스트
dates = df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"].unique() # unique 이용해서 중복 제거

# 데이터 속성 지정
# 행 하나를 뽑아서 저장하고, 두 개 뽑아서 저장하고, 그렇게 반복하여 다 넣는 것
for date in dates:
    frame_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                # date가 돌아갈 변수이므로 하나씩 넣어서 돌아가는 것이다. 그 이하인 것들은 다 끌고가도록
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "marker": {"color": "red"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "line": {"color": "blue", "dash": "dash"}
            }
        ],
        "name": str(date) # 각 frame_data의 이름을 현재 날짜 문자열로 설정
    }
    frames.append(frame_data) # 행 추가
    
frames[3] # 2022-10-03부터 2022-10-06까지의 날짜와 발생건수

# x축과 y축의 범위 설정
x_range = ['2022-10-03', '2023-01-11']
y_range = [8900, 88172]

# 애니메이션을 위한 레이아웃 설정
# updatemenus가 애니메이션(play, pause 두 개로 나눔)
margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
layout = {
    "title": "코로나 19 발생현황",
    "xaxis": {"title": "날짜", "showgrid": False, "range": x_range},
    "yaxis": {"title": "확진자수", "range": y_range},
    "margin": margins_P,
    # Plotly 그래프의 상호작용을 제어하는 버튼 및 드롭다운 메뉴를 정의하는 속성
    "updatemenus": [{
        "type": "buttons",
        "showactive": False, # 시작할 때 시작될지 말지
        "buttons": [{
            "label": "Play",
            "method": "animate", # 버튼을 클릭했을 때 애니메이션을 시작
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
            # args: 애니메이션을 시작할 때 현재 데이터를 사용
            # 각 프레임의 지속 시간(duration)을 500 밀리초로 설정하고, 프레임 변경 시 그래프를 다시 그림(redraw)
            # fromcurrent: 멈췄다가 다시 시작할 때 현재 프레임부터 애니메이션을 시작
        }, {
            "label": "Pause", 
            "method": "animate", # 버튼을 클릭했을 때 애니메이션을 일시 중지
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
            # args: 애니메이션을 멈추기 위해 데이터를 변경하지 않음
            # 애니메이션 프레임의 지속 시간(duration)을 0으로 설정 
            # 프레임 변경 시 그래프를 다시 그리지 않음(redraw)
            # 즉시 애니메이션을 멈춤(immediate)
            # 전환 효과(transition)의 지속 시간(duration)을 0으로 설정하여 전환이 즉시 이루어짐
        }]
    }]
}

# Figure 생성
fig = go.Figure(
    data=[
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout=layout, # layout 할당
    frames=frames # frame 할당
)

fig.show()
