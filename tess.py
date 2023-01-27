import skmob
import geopandas as gpd
from shapely.geometry import Polygon

import utilities as utils
# import folium
# from folium.plugins import HeatMap
# import matplotlib as mpl
import matplotlib.pyplot as plt

from main import capital_df

df_orig = capital_df
df_orig = utils.remove_oomc(df_orig, 'iso 3 code')

df_orig.rename(columns={'iso 3 code': 'tile_ID'}, inplace=True)

# load a spatial tessellation
tessellation = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# rename columns of our tessellation gpf
tessellation.rename(columns={'iso_a3': 'tile_ID'}, inplace=True)
# change iso 3 codes that are not equal as in the given dataset.
#   cases for: France, Norway, Kosovo
tessellation = tessellation
tessellation.loc[tessellation['name'] == 'France', 'tile_ID'] = 'FRA'
tessellation.loc[tessellation['name'] == 'Norway', 'tile_ID'] = 'NOR'

print(tessellation.head())
# add origin and destination ids to tessellation
# tessellation['origin'] = tessellation.loc[:, 'tile_ID']
# tessellation['destination'] = tessellation.loc[:, 'tile_ID']
# tessellation.to_csv('./data/corr_tell.csv', index=False)

""" As the tessellation provided by geopandas and the existing one are different, 
        we will make a comparison between the countries listed on each table """
world_tess = df_orig.merge(tessellation.drop_duplicates(), on=['tile_ID'], how='left', indicator=True)
world_tess = world_tess.dropna()

fdf = skmob.FlowDataFrame.from_file('./data/countries_flows.csv', origin='origin',
                                    destination='destination', tessellation=tessellation, tile_id='tile_ID')

print(fdf.head())

map_f = fdf.plot_tessellation()

x = fdf.plot_flows(map_f=map_f, min_flow=1000, flow_weight=0.5)  # flow_exp=0.5)
x.save(outfile='bebitofiufiu.html')
plt.show()

