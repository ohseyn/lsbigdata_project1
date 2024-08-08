import json

geo = json.load(open("Data/SIG.geojson", encoding="UTF-8"))
geo["features"][0]["properties"] # 행정구역 코드 및 이름(properties)
geo["features"][0]["geometry"] # 위도(latitude), 경도(longitude) 좌표

import pandas as pd

df_pop = pd.read_csv("Data/Population_SIG.csv")
df_pop.head()
df_pop.info()

df_pop["code"] = df_pop["code"].astype(str)

!pip install folium
import folium

geo_map = folium.Map(location = [35.95, 127.7], zoom_start = 8) # 지도 중심 좌표, 확대 단계

#=======================================
# 서울시 지도
geo_seoul = json.load(open("Data/SIG_Seoul.geojson", encoding = "UTF-8"))
type(geo_seoul) # <class 'dict'>
len(geo_seoul) # 4(key 개수)
geo_seoul.keys() # dict_keys(['type', 'name', 'crs', 'features'])

len(geo_seoul["features"]) # 25(아마 서울시 구 개수인 듯)
len(geo_seoul["features"][0]) # 3(키 개수)
type(geo_seoul["features"][0]) # <class 'dict'>

# 숫자가 바뀌면 '구'가 바뀜
geo_seoul["features"][0]["properties"] # 행정구역 코드 및 이름(properties)
# {'SIG_CD': '11110', 'SIG_ENG_NM': 'Jongno-gu', 'SIG_KOR_NM': '종로구'}
geo_seoul["features"][0]["geometry"] # 위도(latitude), 경도(longitude) 좌표 # <class 'dict'>
geo_seoul["features"][0].keys() # dict_keys(['type', 'properties', 'geometry'])

geo_seoul['features'][0]['geometry'].keys() # # coordinates는 좌표 # dict_keys(['type', 'coordinates'])
coordinate_list = geo_seoul['features'][0]['geometry']['coordinates']
len(coordinate_list) # 1, 대괄호 4개
len(coordinate_list[0])  # 1, 대괄호 3개
len(coordinate_list[0][0]) # 2332개, 이제 pd.df로 만들기

import numpy as np

coordinate_array = np.array(coordinate_list[0][0])
# array([[127.00864326,  37.58046825],
#       [127.00871275,  37.58045117],
#       [127.00876564,  37.58044311],
#       ...,
#       [127.00857739,  37.58063571],
#       [127.00858204,  37.58062895],
#       [127.00864326,  37.58046825]])

x = coordinate_array[:,0]
y = coordinate_array[:,1]

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
plt.clf()

# 함수로 만들기
def draw_seoul(num):
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list = geo_seoul['features'][num]['geometry']['coordinates']
    coordinate_array = np.array(coordinate_list[0][0])
    
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    
    plt.rcParams.update({"font.family": "Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
    return None

draw_seoul(19) # 동작구

#===================================================
# 서울시 전체 지도 그리기
geo_mex=[]
geo_mey=[]
geo_name=[]

for x in np.arange(0,25):
    gu_name=geo_seoul["features"][x]["properties"]['SIG_KOR_NM']
    coordinates_list=geo_seoul["features"][x]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinates_list[0][0])
    
    geo_mex.append(coordinate_array[:,0])
    geo_mey.append(coordinate_array[:,1])
    geo_name.append(gu_name)

for x in np.arange(0,25):
    plt.plot(geo_mex[x],geo_mey[x])
    plt.show()
    
plt.clf() 

#=======================================================
# 구이름 만들기
# 1번
gu_name=list()
for i in range(25):
    gu_name.append(geo_seoul["features"][i]["properties"]['SIG_KOR_NM'])
# 2번
gu_name = [geo_seoul["features"][i]["properties"]['SIG_KOR_NM'] for i in range(len(geo_seoul["features"]))]

# x,y 판다스 데이터 프레임

def make_seouldf(num):
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list = geo_seoul['features'][num]['geometry']['coordinates']
    coordinate_array = np.array(coordinate_list[0][0])
    
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    
    return pd.DataFrame({"gu_name": gu_name, "x": x, "y": y})

result=pd.DataFrame({})

for i in range(25):
        result=pd.concat([result, make_seouldf(i)], ignore_index=True)

import seaborn as sns

# result["x"] # 시리즈라서 안 됨
result[["x"]]
sns.scatterplot(data=result, x ="x", y="y", hue="gu_name", palette="viridis", s = 2, legend=False)
plt.show()
plt.clf()

# 동작구만 표시하고 싶을 때
dongjak_df = result.assign(is_dongjak=np.where(result["gu_name"] == "동작구", "동작", "안동작"))
sns.scatterplot(data=dongjak_df, x ="x", y="y", palette={"동작":"red", "안동작": "grey"}, hue="is_dongjak", s = 2)
plt.show()
plt.clf()

#=======================================================

geo_seoul = json.load(open("Data/SIG_Seoul.geojson", encoding = "UTF-8"))
geo_seoul["features"][0]["properties"]

df_pop = pd.read_csv("Data/Population_SIG.csv")
df_seoulpop = df_pop.iloc[1:26]
df_seoulpop["code"] = df_seoulpop["code"].astype(str)
df_seoulpop.info()

import folium

center_x=result["x"].mean()
center_y=result["y"].mean()

# 흰 도화지 맵 가져오기
map_sig=folium.Map(location = [37.551, 126.97], zoom_start=8, tiles="cartodbpositron") # 값 넣을 때 위도, 경도 순
map_sig.save("map_seoul.html")

# 코로플릿로 구 경계선 그리기
geo_seoul["features"][0]["properties"]["SIG_CD"]
folium.Choropleth(
    geo_data=geo,
    data = df_seoulpop,
    columns = ("code", "pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
map_sig.save("map_sig.html")

# 범주 만들어서 그리기
bins = list(df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
folium.Choropleth(
    geo_data=geo,
    data = df_seoulpop,
    columns = ("code", "pop"),
    bins = bins,
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
map_sig.save("map_sig_bins.html")

# 점 찍는 법
make_seouldf(19).iloc[:, 1:3].mean() # 전체 행의 위치를 경도, 위도 좌표를 뽑아서 평균 냄냄
# Marker([marker를 표시해줄 위도, 경도 정보], popup="marker를 클릭했을 때 보여줄 정보").add_to(변수)
folium.Marker([37.495259, 126.942183], popup="동작구").add_to(map_sig) # 그 값을 Marker 값에 넣어서 표시
map_sig.save("map_sig_dongjak.html")
