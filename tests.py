import math
import pandas as pd
from main import capital_df, countries_df
import utilities as utils

print(capital_df.head())
print(countries_df.head())


def distance_between_points(origin_df, origin_lat_col, origin_lon_col,
                            destination_df, destination_lat_col, destination_lon_col):
    R = 6371  # radius of Earth in kilometers
    origin_df['Distance'] = None
    for i, row in origin_df.iterrows():
        lat1 = row[origin_lat_col]
        lon1 = row[origin_lon_col]
        for j, dest_row in destination_df.iterrows():
            lat2 = dest_row[destination_lat_col]
            lon2 = dest_row[destination_lon_col]
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            delta_phi = math.radians(lat2 - lat1)
            delta_lambda = math.radians(lon2 - lon1)
            a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c
            origin_df.at[i, 'Distance'] = distance
    return origin_df


origin_df = pd.DataFrame({'Latitude': [40.730610, 32.715], 'Longitude': [-73.935242, -117.1625]})
destination_df = pd.DataFrame({'Latitude': [41.848, 38.307636666666665], 'Longitude': [-87.6614, -85.90693666666666]})
result = distance_between_points(origin_df, 'Latitude', 'Longitude', destination_df, 'Latitude', 'Longitude')
print(result)

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