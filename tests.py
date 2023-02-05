import math
import pandas as pd
from main import capital_df, countries_df
import utilities as utils

print(capital_df.head())
print(countries_df.head())


def coord_to_dict(df):
    list_key = []
    list_lng = []
    list_lat = []
    for row in df.index:
        list_key.append(df.iloc[row, 'tile_ID'])
        list_lng.append(df.iloc[row, 'lng'])
        list_lat.append(df.iloc[row, 'lat'])
    todict = {'key': list_key, 'lng': list_lng, 'lat': list_lat}
    return todict


def distance_between_points(flow_df, coord_df):
    R = 6371  # radius of Earth in kilometers
    coord_dict = capital_df.to_dict('dict')
    flow_df['Distance'] = None
    """
    for i, row in flow_df.iterrows():
        
        for j, dest_row in destination_df.iterrows():
    """
    return flow_df


print(distance_between_points(countries_df, capital_df))
"""
origin_df = pd.DataFrame({'Latitude': [40.730610, 32.715], 'Longitude': [-73.935242, -117.1625]})
destination_df = pd.DataFrame({'Latitude': [41.848, 38.307636666666665], 'Longitude': [-87.6614, -85.90693666666666]})
result = distance_between_points(origin_df, 'Latitude', 'Longitude', destination_df, 'Latitude', 'Longitude')
print(result)
"""


"""
from geopy.distance import geodesic
import pandas as pd
from main import capital_df, countries_df

print(capital_df.head())
print(countries_df.head())

df_from = capital_df
df_dest = countries_df

df_dest['distance'] = df_dest.apply(lambda x: geodesic((x['lat'],
                                                        x['lng']),
                                                       (df_from[df_dest['iso 3 code'] == x['iso 3 code']]['lat'].values[0],
                                                        df_from[df_dest['iso 3 code'] == x['iso 3 code']]['lng'].values[0])).km, axis=1)
print(df_dest.head())
"""