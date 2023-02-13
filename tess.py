import skmob
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import utilities as utils
# import folium
# from folium.plugins import HeatMap
# import matplotlib as mpl
import matplotlib.pyplot as plt


def add_gdp(tessdf):
    gdp_df = pd.read_csv('./data/out_put.csv', index_col=[0])
    gdp_df.drop(['country', 'Country or Area'], axis=1, inplace=True)
    tessdf = tessdf.merge(gdp_df, on=['tile_ID'], how='left')
    tessdf.rename(columns={'usd_value': 'gdp'}, inplace=True)
    return tessdf


def generate_tessellation(df):
    # load a spatial tessellation
    tess = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # rename columns of our tessellation gpf
    tess.rename(columns={'iso_a3': 'tile_ID'}, inplace=True)
    # change iso 3 codes that are not equal as in the given dataset.
    #   cases for: France, Norway, Kosovo
    tess = tess
    tess.loc[tess['name'] == 'France', 'tile_ID'] = 'FRA'
    tess.loc[tess['name'] == 'Norway', 'tile_ID'] = 'NOR'
    tess.rename(columns={'gdp_md_est': 'gdp'}, inplace=True)
    return tess


def remove_multipolygons(df):
    tess_orig = generate_tessellation(df)
    tess_expl = tess_orig.explode(index_parts=True)
    tess_expl.reset_index(inplace=True)
    tess_expl = tess_expl[tess_expl['level_1'] == 0]
    return tess_expl


def generate_flow_map(fdf):
    map_f = fdf.plot_tessellation()
    x = fdf.plot_flows(map_f=map_f, min_flow=1000, flow_weight=0.5)  # flow_exp=0.5)
    x.save(outfile='oggi.html')
    plt.show()

"""

"""