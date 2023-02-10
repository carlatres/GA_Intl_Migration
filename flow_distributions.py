# %matplotlib inline

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import numpy as np


# Read Datasets
def read_abel_s1():
    df = pd.read_excel("./data/abel-database-s1.xlsx", sheet_name='look up')
    return df


def read_abel_s2():
    df = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2005-10', index_col=0)
    return df


def read_all_years_flows():
    df_1 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='1990-95', index_col=0)
    df_1.reset_index(inplace=True)
    df_1.rename(columns={'index': 'iso 3 code'}, inplace=True)
    df_2 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='1995-2000', index_col=0)
    df_2.reset_index(inplace=True)
    df_2.rename(columns={'index': 'iso 3 code'}, inplace=True)
    df_3 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2000-05', index_col=0)
    df_3.reset_index(inplace=True)
    df_3.rename(columns={'index': 'iso 3 code'}, inplace=True)
    df_4 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2005-10', index_col=0)
    df_4.reset_index(inplace=True)
    df_4.rename(columns={'index': 'iso 3 code'}, inplace=True)
    return df_1, df_2, df_3, df_4


def extract_by_label(df_1, df_2, df_3, df_4, label):
    list_1 = df_1.loc[df_1['iso 3 code'] == label]
    list_2 = df_2.loc[df_2['iso 3 code'] == label]
    list_3 = df_3.loc[df_3['iso 3 code'] == label]
    list_4 = df_4.loc[df_4['iso 3 code'] == label]

    df = pd.concat([list_1, list_2, list_3, list_4])
    df.reset_index(inplace=True, drop=True)

    range_years = ['90-95', '95-00', '00-05', '05-10']
    df['range'] = range_years

    df.drop(['iso 3 code'], axis=1, inplace=True)

    return df


def top_emigration_countries(df):
    df = df.transpose()
    df.sort_values([0], ascending=False, axis=0, inplace=True)

    return df



def graph_top5(df):
    df.sort_values(['Total'], ascending=False, axis=0, inplace=True)
    # get the top 5 entries
    df_top5 = df.head()
    # transpose the dataframe
    df_top5 = df_top5.transpose()

    df_top5.head()
    df_top5.index = df_top5.reset_index  # let's change the index values of df_top5 to type integer for plotting
    df_top5.plot(kind='area',
                 stacked=False,
                 figsize=(20, 10),  # pass a tuple (x, y) size
                 )

    plt.title('Immigration Trend of Top 5 Countries')
    plt.ylabel('Number of Immigrants')
    plt.xlabel('Years')

    plt.show()


df_90, df_95, df_00, df_05 = read_all_years_flows()
df_italy = extract_by_label(df_90, df_95, df_00, df_05, 'ITA')
df_total = extract_by_label(df_90, df_95, df_00, df_05, 'TOTAL')
# graph_top5(df_total)

df_total_2 = top_emigration_countries(df_total)

print(df_05.head())


