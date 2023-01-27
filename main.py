import pandas as pd
import utilities as utils
from data import generate_df, generate_capitals
from geoGeneration import generate_latlon
import skmob
import matplotlib.pyplot as plt
import geopandas as gpd

# from skmob.utils import utils, constants
# from skmob.tessellation import tilers
# from skmob.utils.plot import plot_gdf

# import numpy as np
# import geopandas as gpd
# import shapely
# import folium
# from folium.plugins import HeatMap
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# read countries data and generate dataframe
c_df = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2005-10', index_col=0)
c_df = c_df.iloc[:-1, :-1]
countries_df = generate_df(c_df)

# remove countries out of the map declared by the geopandas low res countries dataset
countries_df = utils.remove_orig_dest_countries(countries_df)
# as we are generating the FlowDataFrame from a file, save our information to a CSV file
#   to retrieve it in the FlowDataFrame generation
countries_df.to_csv('./data/countries_flows.csv', index=False)

"""
# read world regions data and generate dataframe
r_df = pd.read_excel("./data/abel-database-s1.xlsx", sheet_name='flow estimates by region 2005', index_col=0)
regions_df = generate_df(r_df)
regions_df.to_csv('./data/regions_flows.csv', index=False)
"""

# read country and world region look up data
capital_df = pd.read_excel("./data/abel-database-s1.xlsx", sheet_name='look up')
generate_capitals(capital_df)
generate_latlon(capital_df)

# drops empty capitals
capital_df = capital_df.dropna()
capital_df = utils.remove_oomc(capital_df, 'iso 3 code')

capital_df.rename(columns={'iso 3 code': 'tile_ID'}, inplace=True)

print(capital_df.head())

""" This is plot distribution for flow and flow-distance """

