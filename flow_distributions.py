import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def num_index(df):
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'tile_ID'}, inplace=True)
    return df


def read_all_years_flows():
    df_1 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='1990-95', index_col=0)
    df_1.reset_index(inplace=True)
    df_1.rename(columns={'index': 'tile_ID'}, inplace=True)
    df_1.drop(['TOTAL'], axis=1, inplace=True)
    df_2 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='1995-2000', index_col=0)
    df_2.reset_index(inplace=True)
    df_2.rename(columns={'index': 'tile_ID'}, inplace=True)
    df_2.drop(['TOTAL'], axis=1, inplace=True)
    df_3 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2000-05', index_col=0)
    df_3.reset_index(inplace=True)
    df_3.rename(columns={'index': 'tile_ID'}, inplace=True)
    df_3.drop(['TOTAL'], axis=1, inplace=True)
    df_4 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2005-10', index_col=0)
    df_4.reset_index(inplace=True)
    df_4.rename(columns={'iso 3 code': 'tile_ID'}, inplace=True)
    df_4.drop(['TOTAL'], axis=1, inplace=True)
    return df_1, df_2, df_3, df_4


def extract_by_orig_label(df_1, df_2, df_3, df_4, label):
    list_1 = df_1.loc[df_1['tile_ID'] == label]
    list_2 = df_2.loc[df_2['tile_ID'] == label]
    list_3 = df_3.loc[df_3['tile_ID'] == label]
    list_4 = df_4.loc[df_4['tile_ID'] == label]

    df = pd.concat([list_1, list_2, list_3, list_4])
    df.reset_index(inplace=True, drop=True)
    range_years = [1995, 2000, 2005, 2010]
    df['years'] = range_years
    df.drop(['tile_ID'], axis=1, inplace=True)
    return df


def extract_by_dest_label(df_1, df_2, df_3, df_4, label):
    df_1['TOTAL'] = df_1.sum(axis=1)
    df_1 = df_1.transpose()
    df_1 = df_1.rename(columns=df_1.iloc[0]).drop(df_1.index[0])
    df_1 = num_index(df_1)
    df_2['TOTAL'] = df_2.sum(axis=1)
    df_2 = df_2.transpose()
    df_2 = df_2.rename(columns=df_2.iloc[0]).drop(df_2.index[0])
    df_2 = num_index(df_2)
    df_3['TOTAL'] = df_3.sum(axis=1)
    df_3 = df_3.transpose()
    df_3 = df_3.rename(columns=df_3.iloc[0]).drop(df_3.index[0])
    df_3 = num_index(df_3)
    df_4['TOTAL'] = df_4.sum(axis=1)
    df_4 = df_4.transpose()
    df_4 = df_4.rename(columns=df_4.iloc[0]).drop(df_4.index[0])
    df_4 = num_index(df_4)

    list_1 = df_1.loc[df_1['tile_ID'] == label]
    list_2 = df_2.loc[df_2['tile_ID'] == label]
    list_3 = df_3.loc[df_3['tile_ID'] == label]
    list_4 = df_4.loc[df_4['tile_ID'] == label]

    df = pd.concat([list_1, list_2, list_3, list_4])
    df.reset_index(inplace=True, drop=True)
    range_years = [1995, 2000, 2005, 2010]
    df.drop(['TOTAL'], axis=1, inplace=True)
    df['years'] = range_years
    df.drop(['tile_ID'], axis=1, inplace=True)
    return df


def top_migration_countries(df):
    df.head()
    df.set_index('years', inplace=True)
    df = df.transpose()
    df['Total'] = df.sum(axis=1)
    df.sort_values(['Total'], ascending=False, axis=0, inplace=True)
    df_top5 = df.head(10)
    return df_top5


# get flows
def get_orig_flows(df_1, df_2, df_3, df_4, label):
    df = extract_by_orig_label(df_1, df_2, df_3, df_4, label)
    df_top10 = top_migration_countries(df)
    years = ['1995', '2000', '2005', '2010']
    df_top10.drop(['Total'], axis=1, inplace=True)
    df_top10 = df_top10.transpose()
    df_top10.index = df_top10.index.map(int)
    return df_top10


def get_dest_flows(df_1, df_2, df_3, df_4, label):
    df = extract_by_dest_label(df_1, df_2, df_3, df_4, label)
    df_top10 = top_migration_countries(df)
    years = ['1995', '2000', '2005', '2010']
    df_top10.drop(['Total'], axis=1, inplace=True)
    df_top10 = df_top10.transpose()
    df_top10.index = df_top10.index.map(int)
    return df_top10
