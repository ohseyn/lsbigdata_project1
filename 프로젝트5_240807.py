import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

hf_df = pd.read_csv("houseprice-with-lonlat.csv")
hf_df = hf_df.iloc[:, 72:75]

center_x=hf_df["Longitude"].mean()
center_y=hf_df["Latitude"].mean()

hf_map=folium.Map(location = [42.03, -93.64], zoom_start=12, tiles="cartodbpositron")
hf_map.save("hf_map.html")

folium.Marker([42.03, -93.64], popup="AMES").add_to(hf_map)
hf_map.save("hf_map_ames.html")

for i in range(2930):
    folium.Marker([hf_df["Latitude"][i], hf_df["Longitude"][i]], popup=f"Price: ${hf_df['Sale_Price'][i]}").add_to(hf_map)
hf_map.save("hf_map_ames_1.html")

#=================================================
# zip 코드 활용(다른 팀 코드) 
hf_map_team1=folium.Map(location = [42.03, -93.64], zoom_start=12, tiles="cartodbpositron")
for price, lat, lon in zip(hf_df['Sale_Price'], hf_df['Latitude'], hf_df['Longitude']):
    folium.Circle(
        location=[lat, lon],
        radius=50,
        fill=True,
        fill_opacity=0.6,
        popup=f"Price: ${price}"
    ).add_to(hf_map_team1)
hf_map_team1.save("hf_map_team1.html")

#==================================================
from folium.plugins import MarkerCluster
house_df = pd.read_csv("houseprice-with-lonlat.csv")

house_df = house_df[["Longitude", "Latitude", "Sale_Price"]]

center_x=house_df["Longitude"].mean()
center_y=house_df["Latitude"].mean()

map_sig=folium.Map(location = [42.034, -93.642], zoom_start = 12, tiles="cartodbpositron")

# 여기서부터 MarkerCluster 쓴 부분!
marker_cluster = MarkerCluster().add_to(map_sig)

for idx, row in house_df.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"Price: ${price}"
    ).add_to(marker_cluster)

map_sig.save("hf_map_team2.html")
