import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import generate_df, generate_capitals
from flow_distance_distributions import get_top_flow_distances, add_flow_distances, read_abel_s1
from flow_distributions import read_all_years_flows, get_orig_flows, get_dest_flows, extract_by_orig_label, \
    extract_by_dest_label
from geoGeneration import generate_latlon
from tess import generate_tessellation, remove_multipolygons
import utilities

import skmob
from skmob.utils import constants
from sklearn.model_selection import train_test_split
from skmob.measures.evaluation import r_squared, mse, spearman_correlation, \
    pearson_correlation, common_part_of_commuters, common_part_of_commuters_distance
from skmob.models.gravity import Gravity
from skmob.models.radiation import Radiation


""" -------------------------------------------------------------------- """
""" ________________________ Dataset exploration _______________________ """

# read countries data and generate dataframe
c_df = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2005-10', index_col=0)
c_df = c_df.iloc[:-1, :-1]
countries_df = generate_df(c_df)

# remove countries out of the map declared by the geopandas low res countries dataset
countries_df = utilities.remove_orig_dest_countries(countries_df)
countries_df = utilities.remove_orig_dest_countries_tess(countries_df)
# as we are generating the FlowDataFrame from a file, save our information to a CSV file
#   to retrieve it in the FlowDataFrame generation
countries_df.to_csv('./data/countries_flows.csv', index=False)

# read country and world region look up data
capital_df = pd.read_excel("./data/abel-database-s1.xlsx", sheet_name='look up')
generate_capitals(capital_df)
generate_latlon(capital_df)

# drops empty capitals
capital_df = capital_df.dropna()
capital_df = utilities.remove_oomc(capital_df, 'iso 3 code')

capital_df.rename(columns={'iso 3 code': 'tile_ID'}, inplace=True)

""" -------------------------------------------------------------------- """
""" __________________ FlowDataFrames Plot_Flows _______________________ """
""" NOTE.- The plot flow could not be viewed as a plt.show, the file had to be saved as plot_flows_countries.html """

tessellation = generate_tessellation()
fdf = skmob.FlowDataFrame.from_file('./data/countries_flows.csv', origin='origin',
                                    destination='destination', tessellation=tessellation, tile_id='tile_ID')
map_f = fdf.plot_tessellation()
x = fdf.plot_flows(map_f=map_f, min_flow=1000, flow_weight=0.5)  # flow_exp=0.5)
x.save(outfile='plot_flows_countries.html')
plt.show()

""" -------------------------------------------------------------------- """
""" ________ Distribution of Migration Flows and Flow Distances ________ """

# MIGRATION FLOW DISTRIBUTIONS

# base data for flow distance distributions
df_90, df_95, df_00, df_05 = read_all_years_flows()

# top destination for all migrants flows
df_top10_total = get_orig_flows(df_90, df_95, df_00, df_05, 'TOTAL')
df_top10_total.plot(kind='area',
                    stacked=False,
                    figsize=(20, 10))
plt.title('Immigration Trend Destination for All Migrants Flows')
plt.ylabel('Flow of Immigrants')
plt.xlabel('Years')
plt.show()

# top destination for migrants from Italy flows
df_top10_total = get_orig_flows(df_90, df_95, df_00, df_05, 'ITA')
df_top10_total.plot(kind='area',
                    stacked=False,
                    figsize=(20, 10))
plt.title('Immigration Trend Destination for Migrants from Italy Flows')
plt.ylabel('Flow of Immigrants')
plt.xlabel('Years')
plt.show()

# top origin of all migrants flows
df_top10_total = get_dest_flows(df_90, df_95, df_00, df_05, 'TOTAL')
df_top10_total.plot(kind='area',
                    stacked=False,
                    figsize=(20, 10))
plt.title('Top Origin of All Migrants Flows')
plt.ylabel('Flow of Immigrants')
plt.xlabel('Years')
plt.show()

# top origin of migrants to Italy flows
df_top10_total = get_dest_flows(df_90, df_95, df_00, df_05, 'ITA')
df_top10_total.plot(kind='area',
                    stacked=False,
                    figsize=(20, 10))
plt.title('Top Origin of Migrants to Italy Flows')
plt.ylabel('Flow of Immigrants')
plt.xlabel('Years')
plt.show()

# FLOW DISTANCE DISTRIBUTIONS

# base data for flow distance distributions
lat_lng_df = read_abel_s1()
years = ['1995', '2000', '2005', '2010']
df_90, df_95, df_00, df_05 = read_all_years_flows()

# flow distance distribution for migrants from Italy
flow_df = extract_by_orig_label(df_90, df_95, df_00, df_05, 'ITA')
flow_df = add_flow_distances(flow_df, lat_lng_df, 'ITA')

df_top10_total = get_top_flow_distances(flow_df)
df_top10_total.plot(kind='area',
                    stacked=False,
                    figsize=(20, 10))
plt.title('Destination for Top Flow Distance Migrants from Italy Flows')
plt.ylabel('Distance Flow of Immigrants')
plt.xlabel('Years')
plt.show()

# flow distance distribution for migrants from USA
flow_df = extract_by_orig_label(df_90, df_95, df_00, df_05, 'USA')
flow_df = add_flow_distances(flow_df, lat_lng_df, 'USA')

df_top10_total = get_top_flow_distances(flow_df)
df_top10_total.plot(kind='area',
                    stacked=False,
                    figsize=(20, 10))
plt.title('Destination for Top Flow Distance Migrants from USA Flows')
plt.ylabel('Distance Flow of Immigrants')
plt.xlabel('Years')
plt.show()

# flow distance distribution for migrants to Italy
flow_df = extract_by_dest_label(df_90, df_95, df_00, df_05, 'ITA')
flow_df = add_flow_distances(flow_df, lat_lng_df, 'ITA')

df_top10_total = get_top_flow_distances(flow_df)
df_top10_total.plot(kind='area',
                    stacked=False,
                    figsize=(20, 10))
plt.title('Origin of Top Flow Distance Migrants to Italy Flows')
plt.ylabel('Distance Flow of Immigrants')
plt.xlabel('Years')
plt.show()

# flow distance distribution for migrants to USA
flow_df = extract_by_dest_label(df_90, df_95, df_00, df_05, 'USA')
flow_df = add_flow_distances(flow_df, lat_lng_df, 'USA')

df_top10_total = get_top_flow_distances(flow_df)
df_top10_total.plot(kind='area',
                    stacked=False,
                    figsize=(20, 10))
plt.title('Origin of Top Flow Distance Migrants to USA Flows')
plt.ylabel('Distance Flow of Immigrants')
plt.xlabel('Years')
plt.show()


""" -------------------------------------------------------------------- """
""" ___________________________ Gravity Model __________________________ """

c_df = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2005-10', index_col=0)
c_df = c_df.iloc[:-1, :-1]
flow_by_country_df = generate_df(c_df)

df_w_lonlat = pd.read_excel("./data/abel-database-s1.xlsx", sheet_name='look up')
generate_capitals(df_w_lonlat)
generate_latlon(df_w_lonlat)
df_w_lonlat.rename(columns={'iso 3 code': 'tile_ID'}, inplace=True)

tdf = skmob.TrajDataFrame(df_w_lonlat, latitude='latitude', longitude='longitude')
tdf.crs = 'epsg:4326'

tessellation = generate_tessellation()
# tdf mapping does not accept multipolygons, it was needed to transform the multipolygon into polygons.
#  The polygon after this point correspond to the continental part or larger island of the country/region
tessellation_exploded = remove_multipolygons()

# mapping recognizes tile_ID_left and tile_ID_right.
#   Constant value is changed in order to avoid this error with the method
constants.TILE_ID = 'tile_ID_left'
# assign each point to the corresponding tile
mtdf = tdf.mapping(tessellation_exploded, remove_na=True)
mtdf.head(10)
# constans tile ID is returned to the original value
constants.TILE_ID = 'tile_ID'

# compute relevance
relevances = df_w_lonlat.groupby(by='tile_ID').count()[['lat']].rename(columns={'lat': 'relevance'})
relevances /= relevances.sum()  # normalize

tessellation_exploded = tessellation_exploded.merge(relevances, right_index=True, left_on='tile_ID',
                                                    how='left').fillna(0.)
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

if 'tot_outflow' not in tessellation_exploded.columns:
    tessellation_exploded = tessellation_exploded.merge(tot_outflows, right_index=True, left_on='tile_ID',
                                                        how='left').fillna(0.).sort_values(by='tot_outflow',
                                                                                           ascending=False)
tessellation_exploded.head()

# fit the gravity's model parameters
fdf_train.head()
fdf_train = fdf_train.loc[fdf_train.origin != 'TTO']
fdf_train = fdf_train.loc[fdf_train.origin != 'SSD']
fdf_train = fdf_train.loc[fdf_train.destination != 'TTO']
fdf_train = fdf_train.loc[fdf_train.destination != 'SSD']
train_against = fdf_train.loc[fdf_train['flow'] > 10]

""" ------------------- Gravity Model G1 with Population ------------------- """
gravity_singly_fitted_G1 = Gravity(gravity_type='singly constrained', name='model G1')
print(gravity_singly_fitted_G1)

gravity_singly_fitted_G1.fit(train_against, relevance_column='pop_est')
print(gravity_singly_fitted_G1)

np.random.seed(0)
sc_fdf_fitted_G1 = gravity_singly_fitted_G1.generate(tessellation_exploded,
                                                     tile_id_column='tile_ID',
                                                     tot_outflows_column='tot_outflow',
                                                     relevance_column='pop_est', out_format='flows')
sc_fdf_fitted_G1.head(3)

# Qualitative evaluation

# Create a baseline model (without dependence on relevance and distance)
baseline_G1 = Gravity(gravity_type='singly constrained', deterrence_func_args=[0.], destination_exp=0.)

np.random.seed(0)
baseline_fdf_G1 = baseline_G1.generate(tessellation_exploded,
                                       tile_id_column='tile_ID',
                                       tot_outflows_column='tot_outflow',
                                       relevance_column='relevance',
                                       out_format='flows')

# Compare real flows against generated flows
xy_g1 = fdf_test.merge(sc_fdf_fitted_G1, on=['origin', 'destination'])[['flow_x', 'flow_y']].values
xy_baseline_g1 = fdf_test.merge(baseline_fdf_G1, on=['origin', 'destination'])[['flow_x', 'flow_y']].values

plt.plot(xy_g1[:, 0], xy_g1[:, 1], '.', label='Gravity')
plt.plot(xy_baseline_g1[:, 0], xy_baseline_g1[:, 1], '*', alpha=0.5, label='Baseline')
x_g1 = np.logspace(0, np.log10(np.max(xy_g1)))
plt.plot(x_g1, x_g1, '--k')
plt.xlabel('Real flow')
plt.ylabel('Model flow')
plt.legend(loc='upper left')
plt.title('Gravity Model G1 with Population - Baseline')
plt.loglog()
plt.savefig('plot_g1_pop.png')
plt.show()

# Quantitative evaluation metrics
metrics = [r_squared, mse, spearman_correlation, pearson_correlation,
           common_part_of_commuters, common_part_of_commuters_distance]
names = ['r_squared', 'mse', 'spearman_correlation', 'pearson_correlation',
         'common_part_of_commuters', 'common_part_of_commuters_distance']

print('Metric: Gravity Model G1 w Population - Baseline')
print('---------------------------')
for i, metric in enumerate(metrics):
    m = metric(xy_g1[:, 0], xy_g1[:, 1])
    b = metric(xy_baseline_g1[:, 0], xy_baseline_g1[:, 1])
    print("%s:   %s - %s" % (names[i], np.round(m, 3), np.round(b, 3)))

""" ------------------- Gravity Model G2 with Population ------------------- """
# second with exponential
gravity_singly_fitted_G2 = Gravity(gravity_type='singly constrained', deterrence_func_type='exponential',
                                   name='model G2')
print(gravity_singly_fitted_G2)

gravity_singly_fitted_G2.fit(train_against, relevance_column='pop_est')
print(gravity_singly_fitted_G2)

np.random.seed(0)
sc_fdf_fitted_G2 = gravity_singly_fitted_G2.generate(tessellation_exploded,
                                                     tile_id_column='tile_ID',
                                                     tot_outflows_column='tot_outflow',
                                                     relevance_column='pop_est',
                                                     out_format='flows')
sc_fdf_fitted_G2.head(3)

# Qualitative evaluation
# Create a baseline model (without dependence on relevance and distance)
baseline_G2 = Gravity(gravity_type='singly constrained',  deterrence_func_type='exponential',
                      deterrence_func_args=[0.], destination_exp=0.)

np.random.seed(0)
baseline_fdf_G2 = baseline_G2.generate(tessellation_exploded,
                                       tile_id_column='tile_ID',
                                       tot_outflows_column='tot_outflow',
                                       relevance_column='relevance',
                                       out_format='flows')

# Compare real flows against generated flows
xy_g2 = fdf_test.merge(sc_fdf_fitted_G2, on=['origin', 'destination'])[['flow_x', 'flow_y']].values
xy_baseline_g2 = fdf_test.merge(baseline_fdf_G2, on=['origin', 'destination'])[['flow_x', 'flow_y']].values

plt.plot(xy_g2[:, 0], xy_g2[:, 1], '.', label='Gravity')
plt.plot(xy_baseline_g2[:, 0], xy_baseline_g2[:, 1], '*', alpha=0.5, label='Baseline')
x_g2 = np.logspace(0, np.log10(np.max(xy_g2)))
plt.plot(x_g2, x_g2, '--k')
plt.xlabel('Real flow')
plt.ylabel('Model flow')
plt.legend(loc='upper left')
plt.title('Gravity Model G2 with Population - Baseline')
plt.loglog()
plt.savefig('plot_g2_pop.png')
plt.show()

# Quantitative evaluation metrics
metrics = [r_squared, mse, spearman_correlation, pearson_correlation,
           common_part_of_commuters, common_part_of_commuters_distance]
names = ['r_squared', 'mse', 'spearman_correlation', 'pearson_correlation',
         'common_part_of_commuters', 'common_part_of_commuters_distance']

print('Metric: Gravity Model G2 w Population - Baseline')
print('---------------------------')
for i, metric in enumerate(metrics):
    m = metric(xy_g2[:, 0], xy_g2[:, 1])
    b = metric(xy_baseline_g2[:, 0], xy_baseline_g2[:, 1])
    print("%s:   %s - %s" % (names[i], np.round(m, 3), np.round(b, 3)))

""" ------------------- Gravity Model G3 with Population ------------------- """
# second with exponential
gravity_globally_fitted_G3 = Gravity(gravity_type='globally constrained', name='model G3')
print(gravity_globally_fitted_G3)

gravity_globally_fitted_G3.fit(train_against, relevance_column='pop_est')
print(gravity_globally_fitted_G3)

np.random.seed(0)
sc_fdf_fitted_G3 = gravity_globally_fitted_G3.generate(tessellation_exploded,
                                                       tile_id_column='tile_ID',
                                                       tot_outflows_column='tot_outflow',
                                                       relevance_column='pop_est', out_format='flows')
sc_fdf_fitted_G3.head(3)

# Qualitative evaluation
# Create a baseline model (without dependence on relevance and distance)
baseline_G3 = Gravity(gravity_type='globally constrained', deterrence_func_args=[0.], destination_exp=0.)

np.random.seed(0)
baseline_fdf_G3 = baseline_G3.generate(tessellation_exploded,
                                       tile_id_column='tile_ID',
                                       tot_outflows_column='tot_outflow',
                                       relevance_column='relevance',
                                       out_format='flows')

# Compare real flows against generated flows
xy_g3 = fdf_test.merge(sc_fdf_fitted_G3, on=['origin', 'destination'])[['flow_x', 'flow_y']].values
xy_baseline_g3 = fdf_test.merge(baseline_fdf_G3, on=['origin', 'destination'])[['flow_x', 'flow_y']].values

plt.plot(xy_g3[:, 0], xy_g3[:, 1], '.', label='Gravity')
plt.plot(xy_baseline_g3[:, 0], xy_baseline_g3[:, 1], '*', alpha=0.5, label='Baseline')
x_g3 = np.logspace(0, np.log10(np.max(xy_g3)))
plt.plot(x_g3, x_g3, '--k')
plt.xlabel('Real flow')
plt.ylabel('Model flow')
plt.legend(loc='upper left')
plt.title('Gravity Model G3 with Population - Baseline')
plt.loglog()
plt.savefig('plot_g3_pop.png')
plt.show()

# Quantitative evaluation metrics
metrics = [r_squared, mse, spearman_correlation, pearson_correlation,
           common_part_of_commuters, common_part_of_commuters_distance]
names = ['r_squared', 'mse', 'spearman_correlation', 'pearson_correlation',
         'common_part_of_commuters', 'common_part_of_commuters_distance']

print('Metric: Gravity Model G4 with Population - Baseline')
print('---------------------------')
for i, metric in enumerate(metrics):
    m = metric(xy_g3[:, 0], xy_g3[:, 1])
    b = metric(xy_baseline_g3[:, 0], xy_baseline_g3[:, 1])
    print("%s:   %s - %s" % (names[i], np.round(m, 3), np.round(b, 3)))

""" ------------------- Gravity Model G4 with Population ------------------- """
# second with exponential
gravity_globally_fitted_G4 = Gravity(gravity_type='globally constrained',
                                     deterrence_func_type='exponential', name='model G4')
print(gravity_globally_fitted_G4)

gravity_globally_fitted_G4.fit(train_against, relevance_column='pop_est')
print(gravity_globally_fitted_G4)

np.random.seed(0)
sc_fdf_fitted_G4 = gravity_globally_fitted_G4.generate(tessellation_exploded,
                                                       tile_id_column='tile_ID',
                                                       tot_outflows_column='tot_outflow',
                                                       relevance_column='pop_est', out_format='flows')
sc_fdf_fitted_G4.head(3)

# Qualitative evaluation
# Create a baseline model (without dependence on relevance and distance)
baseline_G4 = Gravity(gravity_type='globally constrained', deterrence_func_type='exponential',
                      deterrence_func_args=[0.], destination_exp=0.)

np.random.seed(0)
baseline_fdf_G4 = baseline_G4.generate(tessellation_exploded,
                                       tile_id_column='tile_ID',
                                       tot_outflows_column='tot_outflow',
                                       relevance_column='relevance',
                                       out_format='flows')

# Compare real flows against generated flows
xy_g4 = fdf_test.merge(sc_fdf_fitted_G4, on=['origin', 'destination'])[['flow_x', 'flow_y']].values
xy_baseline_g4 = fdf_test.merge(baseline_fdf_G4, on=['origin', 'destination'])[['flow_x', 'flow_y']].values

plt.plot(xy_g4[:, 0], xy_g4[:, 1], '.', label='Gravity')
plt.plot(xy_baseline_g4[:, 0], xy_baseline_g4[:, 1], '*', alpha=0.5, label='Baseline')
x_g4 = np.logspace(0, np.log10(np.max(xy_g4)))
plt.plot(x_g4, x_g4, '--k')
plt.xlabel('Real flow')
plt.ylabel('Model flow')
plt.legend(loc='upper left')
plt.title('Gravity Model G4 with Population - Baseline')
plt.loglog()
plt.savefig('plot_g4_pop.png')
plt.show()

# Quantitative evaluation metrics
metrics = [r_squared, mse, spearman_correlation, pearson_correlation,
           common_part_of_commuters, common_part_of_commuters_distance]
names = ['r_squared', 'mse', 'spearman_correlation', 'pearson_correlation',
         'common_part_of_commuters', 'common_part_of_commuters_distance']

print('Metric:  Gravity Model G4 w Population - Baseline')
print('---------------------------')
for i, metric in enumerate(metrics):
    m = metric(xy_g4[:, 0], xy_g4[:, 1])
    b = metric(xy_baseline_g4[:, 0], xy_baseline_g4[:, 1])
    print("%s:   %s - %s" % (names[i], np.round(m, 3), np.round(b, 3)))

""" -------------------- Radiation Model with Population --------------------"""
np.random.seed(0)
radiation = Radiation()
rad_flows = radiation.generate(tessellation_exploded, tile_id_column='tile_ID',
                               tot_outflows_column='tot_outflow', relevance_column='pop_est',
                               out_format='flows_sample')
print(rad_flows.head())

""" ------------------- Gravity Model G1 with GDP ------------------- """
gravity_singly_fitted_gdp_G1 = Gravity(gravity_type='singly constrained', name='model G1')
print(gravity_singly_fitted_gdp_G1)

gravity_singly_fitted_gdp_G1.fit(train_against, relevance_column='gdp')
print(gravity_singly_fitted_gdp_G1)

np.random.seed(0)
sc_fdf_fitted_gdp_G1 = gravity_singly_fitted_gdp_G1.generate(tessellation_exploded,
                                                             tile_id_column='tile_ID',
                                                             tot_outflows_column='tot_outflow',
                                                             relevance_column='gdp',
                                                             out_format='flows')
sc_fdf_fitted_gdp_G1.head(3)

# Qualitative evaluation

# Create a baseline model (without dependence on relevance and distance)
baseline_gdp_G1 = Gravity(gravity_type='singly constrained', deterrence_func_args=[0.], destination_exp=0.)

np.random.seed(0)
baseline_fdf_gdp_G1 = baseline_gdp_G1.generate(tessellation_exploded,
                                               tile_id_column='tile_ID',
                                               tot_outflows_column='tot_outflow',
                                               relevance_column='gdp',
                                               out_format='flows')

# Compare real flows against generated flows
xy_gdp_g1 = fdf_test.merge(sc_fdf_fitted_gdp_G1, on=['origin', 'destination'])[['flow_x', 'flow_y']].values
xy_baseline_gdp_g1 = fdf_test.merge(baseline_fdf_gdp_G1, on=['origin', 'destination'])[['flow_x', 'flow_y']].values

plt.plot(xy_gdp_g1[:, 0], xy_gdp_g1[:, 1], '.', label='Gravity')
plt.plot(xy_baseline_gdp_g1[:, 0], xy_baseline_gdp_g1[:, 1], '*', alpha=0.5, label='Baseline')
x_gdp_g1 = np.logspace(0, np.log10(np.max(xy_gdp_g1)))
plt.plot(x_gdp_g1, x_gdp_g1, '--k')
plt.xlabel('Real flow')
plt.ylabel('Model flow')
plt.legend(loc='upper left')
plt.title('Gravity Model G1 with GDP - Baseline')
plt.loglog()
plt.savefig('plot_g1_gdp.png')
plt.show()

# Quantitative evaluation metrics
metrics = [r_squared, mse, spearman_correlation, pearson_correlation,
           common_part_of_commuters, common_part_of_commuters_distance]
names = ['r_squared', 'mse', 'spearman_correlation', 'pearson_correlation',
         'common_part_of_commuters', 'common_part_of_commuters_distance']

print('Metric: Gravity Model G1 w GDP - Baseline')
print('---------------------------')
for i, metric in enumerate(metrics):
    m = metric(xy_gdp_g1[:, 0], xy_gdp_g1[:, 1])
    b = metric(xy_baseline_gdp_g1[:, 0], xy_baseline_gdp_g1[:, 1])
    print("%s:   %s - %s" % (names[i], np.round(m, 3), np.round(b, 3)))

""" ------------------- Gravity Model G2 with GDP ------------------- """
# second with exponential
gravity_singly_fitted_gdp_G2 = Gravity(gravity_type='singly constrained', deterrence_func_type='exponential',
                                       name='model G2')
print(gravity_singly_fitted_gdp_G2)

gravity_singly_fitted_gdp_G2.fit(train_against, relevance_column='gdp')
print(gravity_singly_fitted_gdp_G2)

np.random.seed(0)
sc_fdf_fitted_gdp_G2 = gravity_singly_fitted_gdp_G2.generate(tessellation_exploded,
                                                             tile_id_column='tile_ID',
                                                             tot_outflows_column='tot_outflow',
                                                             relevance_column='gdp',
                                                             out_format='flows')
sc_fdf_fitted_gdp_G2.head(3)

# Qualitative evaluation
# Create a baseline model (without dependence on relevance and distance)
baseline_gdp_G2 = Gravity(gravity_type='singly constrained',  deterrence_func_type='exponential',
                          deterrence_func_args=[0.], destination_exp=0.)

np.random.seed(0)
baseline_fdf_gdp_G2 = baseline_gdp_G2.generate(tessellation_exploded,
                                               tile_id_column='tile_ID',
                                               tot_outflows_column='tot_outflow',
                                               relevance_column='gdp',
                                               out_format='flows')

# Compare real flows against generated flows
xy_gdp_g2 = fdf_test.merge(sc_fdf_fitted_gdp_G2, on=['origin', 'destination'])[['flow_x', 'flow_y']].values
xy_baseline_gdp_g2 = fdf_test.merge(baseline_fdf_gdp_G2, on=['origin', 'destination'])[['flow_x', 'flow_y']].values

plt.plot(xy_gdp_g2[:, 0], xy_gdp_g2[:, 1], '.', label='Gravity')
plt.plot(xy_baseline_gdp_g2[:, 0], xy_baseline_gdp_g2[:, 1], '*', alpha=0.5, label='Baseline')
x_gdp_g2 = np.logspace(0, np.log10(np.max(xy_gdp_g2)))
plt.plot(x_gdp_g2, x_gdp_g2, '--k')
plt.xlabel('Real flow')
plt.ylabel('Model flow')
plt.legend(loc='upper left')
plt.title('Gravity Model G2 with GDP')
plt.loglog()
plt.savefig('plot_g2_gdp.png')
plt.show()

# Quantitative evaluation metrics
metrics = [r_squared, mse, spearman_correlation, pearson_correlation,
           common_part_of_commuters, common_part_of_commuters_distance]
names = ['r_squared', 'mse', 'spearman_correlation', 'pearson_correlation',
         'common_part_of_commuters', 'common_part_of_commuters_distance']

print('Metric:  Gravity G2 - Baseline')
print('---------------------------')
for i, metric in enumerate(metrics):
    m = metric(xy_gdp_g2[:, 0], xy_gdp_g2[:, 1])
    b = metric(xy_baseline_gdp_g2[:, 0], xy_baseline_gdp_g2[:, 1])
    print("%s:   %s - %s" % (names[i], np.round(m, 3), np.round(b, 3)))

""" ------------------- Gravity Model G3 with GDP ------------------- """
# second with exponential
gravity_globally_fitted_gdp_G3 = Gravity(gravity_type='globally constrained', name='model G3')
print(gravity_globally_fitted_gdp_G3)

gravity_globally_fitted_gdp_G3.fit(train_against, relevance_column='gdp')
print(gravity_globally_fitted_gdp_G3)

np.random.seed(0)
sc_fdf_fitted_gdp_G3 = gravity_globally_fitted_gdp_G3.generate(tessellation_exploded,
                                                               tile_id_column='tile_ID',
                                                               tot_outflows_column='tot_outflow',
                                                               relevance_column='gdp', out_format='flows')
sc_fdf_fitted_gdp_G3.head(3)

# Qualitative evaluation
# Create a baseline model (without dependence on relevance and distance)
baseline_gdp_G3 = Gravity(gravity_type='globally constrained', deterrence_func_args=[0.], destination_exp=0.)

np.random.seed(0)
baseline_fdf_gdp_G3 = baseline_gdp_G3.generate(tessellation_exploded,
                                               tile_id_column='tile_ID',
                                               tot_outflows_column='tot_outflow',
                                               relevance_column='gdp',
                                               out_format='flows')

# Compare real flows against generated flows
xy_gdp_g3 = fdf_test.merge(sc_fdf_fitted_gdp_G3, on=['origin', 'destination'])[['flow_x', 'flow_y']].values
xy_baseline_gdp_g3 = fdf_test.merge(baseline_fdf_gdp_G3, on=['origin', 'destination'])[['flow_x', 'flow_y']].values

plt.plot(xy_gdp_g3[:, 0], xy_gdp_g3[:, 1], '.', label='Gravity')
plt.plot(xy_baseline_gdp_g3[:, 0], xy_baseline_gdp_g3[:, 1], '*', alpha=0.5, label='Baseline')
x_gdp_g3 = np.logspace(0, np.log10(np.max(xy_gdp_g3)))
plt.plot(x_gdp_g3, x_gdp_g3, '--k')
plt.xlabel('Real flow')
plt.ylabel('Model flow')
plt.legend(loc='upper left')
plt.title('Gravity Model G3 with GDP - Baseline')
plt.loglog()
plt.savefig('plot_g3_gdp.png')
plt.show()

# Quantitative evaluation metrics
metrics = [r_squared, mse, spearman_correlation, pearson_correlation,
           common_part_of_commuters, common_part_of_commuters_distance]
names = ['r_squared', 'mse', 'spearman_correlation', 'pearson_correlation',
         'common_part_of_commuters', 'common_part_of_commuters_distance']

print('Metric:  Gravity Model G3 with GDP - Baseline')
print('---------------------------')
for i, metric in enumerate(metrics):
    m = metric(xy_gdp_g3[:, 0], xy_gdp_g3[:, 1])
    b = metric(xy_baseline_gdp_g3[:, 0], xy_baseline_gdp_g3[:, 1])
    print("%s:   %s - %s" % (names[i], np.round(m, 3), np.round(b, 3)))

""" ------------------- Gravity Model G4 with GDP  ------------------- """
# second with exponential
gravity_globally_fitted_gdp_G4 = Gravity(gravity_type='globally constrained',
                                         deterrence_func_type='exponential', name='model G4')
print(gravity_globally_fitted_gdp_G4)

gravity_globally_fitted_gdp_G4.fit(train_against, relevance_column='gdp')
print(gravity_globally_fitted_gdp_G4)

np.random.seed(0)
sc_fdf_fitted_gdp_G4 = gravity_globally_fitted_gdp_G4.generate(tessellation_exploded,
                                                               tile_id_column='tile_ID',
                                                               tot_outflows_column='tot_outflow',
                                                               relevance_column='gdp',
                                                               out_format='flows')
sc_fdf_fitted_gdp_G4.head(3)

# Qualitative evaluation
# Create a baseline model (without dependence on relevance and distance)
baseline_gdp_G4 = Gravity(gravity_type='globally constrained', deterrence_func_type='exponential',
                          deterrence_func_args=[0.], destination_exp=0.)

np.random.seed(0)
baseline_fdf_gdp_G4 = baseline_gdp_G4.generate(tessellation_exploded,
                                               tile_id_column='tile_ID',
                                               tot_outflows_column='tot_outflow',
                                               relevance_column='gdp',
                                               out_format='flows')

# Compare real flows against generated flows
xy_gdp_g4 = fdf_test.merge(sc_fdf_fitted_gdp_G4, on=['origin', 'destination'])[['flow_x', 'flow_y']].values
xy_baseline_gdp_g4 = fdf_test.merge(baseline_fdf_gdp_G4, on=['origin', 'destination'])[['flow_x', 'flow_y']].values

plt.plot(xy_gdp_g4[:, 0], xy_gdp_g4[:, 1], '.', label='Gravity')
plt.plot(xy_baseline_gdp_g4[:, 0], xy_baseline_gdp_g4[:, 1], '*', alpha=0.5, label='Baseline')
x_gdp_g4 = np.logspace(0, np.log10(np.max(xy_gdp_g4)))
plt.plot(x_gdp_g4, x_gdp_g4, '--k')
plt.xlabel('Real flow')
plt.ylabel('Model flow')
plt.legend(loc='upper left')
plt.title('Gravity Model G4 with GDP - Baseline')
plt.loglog()
plt.savefig('plot_g4_gpd.png')
plt.show()

# Quantitative evaluation metrics
metrics = [r_squared, mse, spearman_correlation, pearson_correlation,
           common_part_of_commuters, common_part_of_commuters_distance]
names = ['r_squared', 'mse', 'spearman_correlation', 'pearson_correlation',
         'common_part_of_commuters', 'common_part_of_commuters_distance']

print('Metric:  Gravity Model G4 with GDP - Baseline')
print('---------------------------')
for i, metric in enumerate(metrics):
    m = metric(xy_gdp_g4[:, 0], xy_gdp_g4[:, 1])
    b = metric(xy_baseline_gdp_g4[:, 0], xy_baseline_gdp_g4[:, 1])
    print("%s:   %s - %s" % (names[i], np.round(m, 3), np.round(b, 3)))

""" -------------------- Radiation Model with Population --------------------"""
np.random.seed(0)
radiation = Radiation()
rad_flows = radiation.generate(tessellation_exploded, tile_id_column='tile_ID',
                               tot_outflows_column='tot_outflow', relevance_column='gdp',
                               out_format='flows_sample')
print(rad_flows.head())
