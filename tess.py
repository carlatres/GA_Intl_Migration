import skmob
import geopandas as gpd
from shapely.geometry import Polygon

import utilities as utils
# import folium
# from folium.plugins import HeatMap
# import matplotlib as mpl
import matplotlib.pyplot as plt

from main import capital_df


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
    # print(tessellation.head())
    """ As the tessellation provided by geopandas and the existing one are different, 
    we will make a comparison between the countries listed on each table """
    world_tess = capital_df.merge(tess.drop_duplicates(), on=['tile_ID'], how='left', indicator=True)
    world_tess = world_tess.dropna()
    return tess


def generate_flow_map(fdf):
    map_f = fdf.plot_tessellation()
    x = fdf.plot_flows(map_f=map_f, min_flow=1000, flow_weight=0.5)  # flow_exp=0.5)
    x.save(outfile='oggi.html')
    plt.show()


tessellation = generate_tessellation(capital_df)
fdf = skmob.FlowDataFrame.from_file('./data/countries_flows.csv', origin='origin',
                                    destination='destination', tessellation=tessellation, tile_id='tile_ID')
# generate_flow_map(fdf)


print(fdf.head())
