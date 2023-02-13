import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from data import generate_df, generate_capitals
from geoGeneration import generate_latlon
from tess import generate_tessellation, remove_multipolygons
import utilities

import skmob
from skmob.utils import utils, constants
from skmob.tessellation import tilers
from skmob.utils.plot import plot_gdf
from sklearn.model_selection import train_test_split
from skmob.measures.evaluation import r_squared, mse, spearman_correlation, \
    pearson_correlation, common_part_of_commuters, common_part_of_commuters_distance
from skmob.models.gravity import Gravity


""" Dataset exploration """
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
capital_df = utilities.remove_oomc(capital_df, 'iso 3 code')

capital_df.rename(columns={'iso 3 code': 'tile_ID'}, inplace=True)
# capital_df.to_csv('./data/countries.csv', index=False)

""" Creating FlowDataFrames and usage of plot_flows """
tessellation = generate_tessellation(capital_df)
fdf = skmob.FlowDataFrame.from_file('./data/countries_flows.csv', origin='origin',
                                    destination='destination', tessellation=tessellation, tile_id='tile_ID')
map_f = fdf.plot_tessellation()
x = fdf.plot_flows(map_f=map_f, min_flow=1000, flow_weight=0.5)  # flow_exp=0.5)
x.save(outfile='plot_flows_countries.html')
plt.show()

print(fdf.head())

""" Distribution of Migration Flows and Flow Distances """


""" Gravity Model """
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
# tot_outflows = tot_outflows[tot_outflows['tot_outflow'] != 0]

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
# train_against = fdf_train.sort_values(by='flow', ascending=False).drop_duplicates(subset=['origin'])
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

"""
my_plot = sc_fdf_fitted.plot_flows(min_flow=20000, tiles='cartodbpositron', flow_weight=2, opacity=0.25)
my_plot.save(outfile='fitted.html')
plt.show()
"""
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
plt.savefig('./plots/plot_g1_pop.png')
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
plt.savefig('./plots/plot_g2_pop.png')
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
plt.savefig('./plots/plot_g3_pop.png')
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
plt.savefig('./plots/plot_g4_pop.png')
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
plt.savefig('./plots/plot_g1_gdp.png')
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
plt.savefig('./plots/plot_g2_gdp.png')
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
plt.savefig('./plots/plot_g3_gdp.png')
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
plt.savefig('./plots/plot_g4_gpd.png')
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

print("llegue aqui")
""""""
