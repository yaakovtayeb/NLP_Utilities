import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pyproj import Transformer
import math

pd.set_option('display.max_columns', None)

data = pd.read_clipboard(sep='\t')
city = pd.read_clipboard(sep=',')
city = city.values.flatten().tolist()
city = [ci.strip() for ci in city]
results = dict()


def find_city(text):
    for c in city:
        if c in text:
            return c
    return np.nan


for r in data.itertuples():
    selected_city = find_city(' '.join(r.text.split(',')[1:]))
    results[selected_city] = results.get(selected_city, 0) + 1


data = pd.DataFrame({'city': list(results.keys()),
                     'number': list(results.values())}).sort_values('number', ascending=False)

new_data = pd.merge(data, places[['lat', 'long', 'city']], left_on='city', right_on='city')

new_data[['number', 'long', 'lat']]


import folium
from folium import plugins
map = folium.Map(location=[30, 37], tiles="cartodbpositron", zoom_start=8)
for row in new_data.itertuples():
    folium.CircleMarker(location=[row.lat, row.long],
                        radius=1,
                        weight=row.number,
                        opacity=0.5).add_to(map)

map.show_in_browser()
map.save("map_.html")
# folium.CircleMarker(location=[new_data.iloc[0]['lat'], new_data.iloc[0]['long']],
#                         radius=1,
#                         weight=10,
#                         opacity=0.5).add_to(map)

# transformer = Transformer.from_crs("epsg:2039", "epsg:4326", always_xy=True)
# places['long'], places['lat'] = transformer.transform(places['x'].values, places['y'].values)


