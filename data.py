import pandas as pd
# import numpy as np
from countryinfo import CountryInfo
# from geopy.exc import GeocoderTimedOut
# from geopy.geocoders import Nominatim


# Read Datasets
def read_abel_s1():
    df = pd.read_excel("./data/abel-database-s1.xlsx", sheet_name='look up')
    return df


def read_abel_s2():
    df = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2005-10', index_col=0)
    return df


def read_flow_allyears():
    df_1 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='1990-95')
    df_2 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='1995-00', index_col=0)
    df_3 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2000-05', index_col=0)
    df_4 = pd.read_excel("./data/abel-database-s2.xlsx", sheet_name='2005-10', index_col=0)
    return df_1, df_2, df_3, df_4


# DataFrame generation for countries and world regions
def generate_df(df):

    # stack the prescribed level from columns to index
    df = df.stack()

    # copy origin/destination from index to lists
    list_origin = []
    list_destination = []
    for row in df.index:
        # print(row, end=" ")
        orig, dest = row
        list_origin.append(orig)
        list_destination.append(dest)
    # add lists to dict
    todict = {'origin': list_origin, 'destination': list_destination, 'flow': df.values.tolist()}
    # change dictionary to a dataframe
    new_df = pd.DataFrame(todict)
    return new_df


def generate_capitals(df):
    country_list = df['country']
    list_capitals = []
    for ctry in country_list:
        try:
            capital = CountryInfo(ctry).capital()
            list_capitals.append(capital)
        except Exception as e:
            capital = 'NA'
            list_capitals.append(capital)
    df['capital'] = list_capitals
    return df
