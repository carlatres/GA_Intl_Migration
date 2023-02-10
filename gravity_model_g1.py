from data import generate_df, generate_capitals
from geoGeneration import generate_latlon
from tess import generate_tessellation
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

tdf = skmob.TrajDataFrame(df_w_lonlat, latitude='latitude', longitude='longitude')

tessellation = generate_tessellation(df_w_lonlat)
tess_df = pd.DataFrame(tessellation)

# assign each point to the corresponding tile
tdf_tid = tdf.mapping(tessellation, remove_na=True)
tdf_tid.head(10)

print(flow_by_country_df)
