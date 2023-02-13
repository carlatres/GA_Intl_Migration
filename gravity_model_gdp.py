from data import generate_df, generate_capitals
from geoGeneration import generate_latlon
from tess import generate_tessellation, remove_multipolygons

import utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skmob
from skmob.utils import constants
from sklearn.model_selection import train_test_split
from skmob.measures.evaluation import r_squared, mse, spearman_correlation, \
    pearson_correlation, common_part_of_commuters, common_part_of_commuters_distance
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
# tessellation_exploded = add_gdp(tessellation_exploded)

# assign each point to the corresponding tile
constants.TILE_ID = 'tile_ID_left'
mtdf = tdf.mapping(tessellation_exploded, remove_na=True)
mtdf.head(10)
constants.TILE_ID = 'tile_ID'

# compute relevance
relevances = df_w_lonlat.groupby(by='tile_ID').count()[['lat']].rename(columns={'lat': 'relevance'})
relevances /= relevances.sum()  # normalize

tessellation_exploded = tessellation_exploded.merge(relevances, right_index=True, left_on='tile_ID',
                                                    how='left').fillna(0.)
tessellation_exploded.head(3)
tessellation_exploded = tessellation_exploded .fillna(0)

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

""" ------------------- MODEL G1  ------------------- """
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
plt.loglog()
plt.savefig('my_plot.png')
plt.show()

# Quantitative evaluation metrics
metrics = [r_squared, mse, spearman_correlation, pearson_correlation,
           common_part_of_commuters, common_part_of_commuters_distance]
names = ['r_squared', 'mse', 'spearman_correlation', 'pearson_correlation',
         'common_part_of_commuters', 'common_part_of_commuters_distance']

print('Metric:  Gravity G1 - Baseline')
print('---------------------------')
for i, metric in enumerate(metrics):
    m = metric(xy_gdp_g1[:, 0], xy_gdp_g1[:, 1])
    b = metric(xy_baseline_gdp_g1[:, 0], xy_baseline_gdp_g1[:, 1])
    print("%s:   %s - %s" % (names[i], np.round(m, 3), np.round(b, 3)))

""" ------------------- MODEL G2  ------------------- """
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
plt.loglog()
plt.savefig('my_plot_g2.png')
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

""" ------------------- MODEL G3  ------------------- """
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
plt.loglog()
plt.savefig('my_plot_g2.png')
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

""" ------------------- MODEL G4  ------------------- """
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
plt.loglog()
plt.savefig('my_plot_g2.png')
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
