import json

geo = json.load(open("Data/SIG.geojson", encoding="UTF-8"))
geo["features"][0]["properties"] # 행정구역 코드
geo["features"][0]["geometry"] # 위도, 경도 좌표

import pandas as pd

df_pop = pd.read_csv("Data/Population_SIG.csv")
df_pop.head()
df_pop.info()

df_pop["code"] = df_pop["code"].astype(str)

!pip install folium
import folium

folium.Map(location = [35.95, 127.7], zoom_start = 8) # 지도 중심 좌표, 확대 단계
