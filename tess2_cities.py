import geopandas as gpd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats

from main import capital_df, countries_df


def get_country_flows(df, country):
    country_mig = df.loc[(df['origin'] == country) | (df['destination'] == country)]
    country_mig = country_mig.reset_index()
    country_mig = country_mig.drop(columns=['index'])
    return country_mig


def total_flows_country(df, country):
    for row in df.index:
        if df.loc[row]['origin'] == country:
            df_country.at[row, 'destination'] = 'OTR'
        else:
            df_country.at[row, 'origin'] = 'OTR'
    return df


centroids_df = capital_df
flow_df = countries_df

df_country = get_country_flows(flow_df, 'ITA')
df_country = total_flows_country(df_country, 'ITA')
df_country.head()

# plt.xscale("log")
g = sns.displot(data=df_country, x="flow", kind="kde",  hue="origin")
# plt.xscale('log')# sns.pairplot(df_country, hue='origin')
plt.show()


"""
mybins = np.logspace(0, np.log(100), 100)

g = sns.JointGrid(df_country, xlim=[.5, 1000000], ylim=[.1, 10000000])
g.plot_marginals(sns.distplot, bins=mybins)
g = g.plot(sns.regplot, sns.distplot)
g = g.annotate(stats.pearsonr)

ax = g.ax_joint
ax.set_xscale('log')

g.ax_marg_x.set_xscale('log')
"""
