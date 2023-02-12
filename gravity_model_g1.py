from data import generate_df, generate_capitals
from geoGeneration import generate_latlon
from tess import generate_tessellation, remove_multipolygons
import utilities
import skmob
from skmob.utils import utils, constants
from skmob.tessellation import tilers
from skmob.utils.plot import plot_gdf
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import folium

from folium.plugins import HeatMap
import matplotlib as mpl
import matplotlib.pyplot as plt
from skmob.models.gravity import Gravity

c_df = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2005-10', index_col=0)
c_df = c_df.iloc[:-1, :-1]
flow_by_country_df = generate_df(c_df)

df_w_lonlat = pd.read_excel("./data/abel-database-s1.xlsx", sheet_name='look up')
generate_capitals(df_w_lonlat)
generate_latlon(df_w_lonlat)
df_w_lonlat.rename(columns={'iso 3 code': 'tile_ID'}, inplace=True)

tdf = skmob.TrajDataFrame(df_w_lonlat, latitude='latitude', longitude='longitude')
tdf.crs = 'epsg:4326'

tessellation = generate_tessellation(df_w_lonlat)
tessellation_exploded = remove_multipolygons(df_w_lonlat)
# tessellation_exploded.set_index('tile_ID', drop=False, inplace=True)
# tess_df = pd.DataFrame(tessellation_exploded)

# assign each point to the corresponding tile
constants.TILE_ID = 'tile_ID_left'
mtdf = tdf.mapping(tessellation_exploded, remove_na=True)
# mtdf = tdf.mapping(tessellation[1:3])
mtdf.head(10)
constants.TILE_ID = 'tile_ID'
# compute relevance
relevances = df_w_lonlat.groupby(by='tile_ID').count()[['lat']].rename(columns={'lat': 'relevance'})
relevances /= relevances.sum()  # normalize

tessellation_exploded = tessellation_exploded.merge(relevances, right_index=True, left_on='tile_ID', how='left').fillna(0.)
tessellation_exploded.head(3)

tessellation = utilities.remove_countries(tessellation, 'tile_ID')

w_tess = tessellation_exploded.merge(df_w_lonlat.drop_duplicates(), on=['tile_ID'], how='left', indicator=True)

# redo FlowDataDrame after changes in tessellation
fdf = skmob.FlowDataFrame.from_file('./data/countries_flows.csv', origin='origin',
                                    destination='destination', tessellation=tessellation_exploded, tile_id='tile_ID')

# split dataset into training and testing
fdf_train, fdf_test = train_test_split(fdf, test_size=0.3, random_state=25)

# compute number of trips for each tile
tot_outflows = fdf_train[fdf_train['origin'] != fdf_train['destination']] \
    .groupby(by='origin', axis=0)[['flow']].sum().fillna(0).rename(columns={'flow': 'tot_outflow'})
# tot_outflows = tot_outflows[tot_outflows['tot_outflow'] != 0]

if 'tot_outflow' not in tessellation_exploded.columns:
    tessellation_exploded = tessellation_exploded.merge(tot_outflows, right_index=True, left_on='tile_ID',
                                      how='left').fillna(0.).sort_values(by='tot_outflow', ascending=False)
tessellation_exploded.head()

# fit the gravity's model parameters
fdf_train.head()

gravity_singly_fitted = Gravity(gravity_type='singly constrained')
print(gravity_singly_fitted)

fdf_train = fdf_train.loc[fdf_train.origin != 'TTO']
fdf_train = fdf_train.loc[fdf_train.origin != 'SSD']
fdf_train = fdf_train.loc[fdf_train.destination != 'TTO']
fdf_train = fdf_train.loc[fdf_train.destination != 'SSD']
# train_against = fdf_train.sort_values(by='flow', ascending=False).drop_duplicates(subset=['origin'])
train_against = fdf_train.loc[fdf_train['flow'] > 10]
gravity_singly_fitted.fit(train_against, relevance_column='pop_est')
print(gravity_singly_fitted)

np.random.seed(0)
sc_fdf_fitted = gravity_singly_fitted.generate(tessellation_exploded,
                tile_id_column='tile_ID',
                tot_outflows_column='tot_outflow',
                relevance_column= 'pop_est', out_format='flows')
sc_fdf_fitted.head(3)

"""
my_plot = sc_fdf_fitted.plot_flows(min_flow=1000, zoom=100, tiles='cartodbpositron', flow_weight=2, opacity=0.25)
my_plot.save(outfile='fitted.html')
plt.show()
"""
# quali

