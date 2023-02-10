from tess import tessellation
from main import capital_df
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
import utilities as utils

print(capital_df)

# tdf = skmob.TrajDataFrame(capital_df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user')
# tdf_tid = tdf.mapping(tessellation, remove_na=True)

# compute relevance
relevances = capital_df.groupby(by='tile_ID').count()[['lat']].rename(columns={'lat': 'relevance'})
relevances /= relevances.sum()  # normalize

tessellation = tessellation.merge(relevances, right_index=True, left_on='tile_ID', how='left').fillna(0.)
tessellation.head(3)

tessellation = utils.remove_countries(tessellation, 'tile_ID')

w_tess = tessellation.merge(capital_df.drop_duplicates(), on=['tile_ID'], how='left', indicator=True)

# redo FlowDataDrame after changes in tessellation
fdf = skmob.FlowDataFrame.from_file('./data/countries_flows.csv', origin='origin',
                                    destination='destination', tessellation=tessellation, tile_id='tile_ID')

# split dataset into training and testing
fdf_train, fdf_test = train_test_split(fdf, test_size=0.3, random_state=25)

# compute number of trips for each tile
tot_outflows = fdf_train[fdf_train['origin'] != fdf_train['destination']] \
    .groupby(by='origin', axis=0)[['flow']].sum().fillna(0).rename(columns={'flow': 'tot_outflow'})
# tot_outflows = tot_outflows[tot_outflows['tot_outflow'] != 0]

if 'tot_outflow' not in tessellation.columns:
    tessellation = tessellation.merge(tot_outflows, right_index=True, left_on='tile_ID',
                                      how='left').fillna(0.).sort_values(by='tot_outflow', ascending=False)
tessellation.head()

# fit the gravity's model parameters
fdf_train.head()

gravity_singly_fitted = Gravity(gravity_type='singly constrained', name='model G1')
print(gravity_singly_fitted)

gravity_singly_fitted.fit(fdf_train, relevance_column='relevance')
print(gravity_singly_fitted)



"""
# total outflows excluding self loops in San Francisco
tot_outflows = fdf_train[fdf_train['origin'] != fdf_train['destination']] \
    .groupby(by='origin', axis=0)[['flow']].sum().fillna(0).rename(columns={'flow': 'tot_outflow'})
    
tot_outflows = fdf[fdf['origin'] != fdf['destination']].groupby(by='origin', axis=0)[['flow']].sum().fillna(0)
tessellation = tessellation.merge(tot_outflows, left_on='tile_ID', right_on='origin').rename(columns={'flow': constants.TOT_OUTFLOW})

print(tessellation.head())
gravity_singly = Gravity(gravity_type='singly constrained')
print(gravity_singly)
"""
