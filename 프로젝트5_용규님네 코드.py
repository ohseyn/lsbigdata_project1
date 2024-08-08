import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap
import pandas as pd

data = pd.read_csv("houseprice-with-lonlat.csv")

data = data.iloc[:, 1:]

# 집 가격 범위랑 그에 따른 색 list 만들기 
price_ranges = [100000, 200000, 300000, 400000, 500000, 600000, 700000, float('inf')]
colors = ['grey', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'black']

# 가격에 따른 색 지정 
def get_color(price):
    for i, range_upper in enumerate(price_ranges):
        if price <= range_upper:
            return colors[i]
        
map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
my_map = folium.Map(location=map_center, zoom_start=12,tiles = "cartodbpositron")

for price, lat, lon in zip(data['Sale_Price'], data['Latitude'], data['Longitude']):
    color = get_color(price)
    folium.Circle(
        location=[lat, lon],
        radius=20,
        color=color,
        popup=f"Price: ${price}",
        fill=True,
        fill_opacity=0.6
    ).add_to(my_map)
my_map.save("hf_map_team3_1.html")

#===================================================

data = pd.read_csv("houseprice-with-lonlat.csv")

map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
my_map = folium.Map(location=map_center, zoom_start=12)

# 가격에 따라 icon 사이즈 조절하기 위하여
def get_icon_size(price):
    return [price / 10000, price / 10000]
#ZIP 안 쓰고
for i in range(len(data)):
    price = data['Sale_Price'][i] # i가 데이터 길이만큼 값을 하나씩 넣어줌
    lat = data['Latitude'][i]
    lon = data['Longitude'][i]
    icon_size = get_icon_size(price) # 비쌀수록 아이콘 크기가 클 것
    
    # 마커 아이콘을 HTML로 정의
    # Font Awesome의 "home" 아이콘을 사용(Font Awesome: 다양한 아이콘을 제공하는 라이브러리)
    # style: 아이콘의 색상과 크기를 설정
    # 아이콘 사이즈는 가격에 따라 크기가 달라짐{icon_size[0]}
    icon_html = f'<i class="fa fa-home" style="color:#FF1234;font-size:{icon_size[0]}px;"></i>'
    # 아이콘 클래스
    # HTML과 CSS를 사용하여 커스터마이즈된 아이콘을 지도에 추가
    # icon_html 문자열을 아이콘의 HTML로 사용, 아이콘이 렌더링
    icon = folium.DivIcon(html=icon_html)
    
    folium.Marker(
        location=[lat, lon],
        popup=f"Price: ${price}",
        icon=icon
    ).add_to(my_map)
my_map.save("hf_map_team3_2.html")
