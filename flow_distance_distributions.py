import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import utilities
import numpy as np
import math

from data import generate_capitals
from flow_distributions import extract_by_orig_label, read_all_years_flows, extract_by_dest_label
from geoGeneration import generate_latlon


def read_abel_s1():
    cap_df = pd.read_excel("./data/abel-database-s1.xlsx", sheet_name='look up')
    generate_capitals(cap_df)
    generate_latlon(cap_df)
    # drops empty capitals
    cap_df = cap_df.dropna()
    cap_df = utilities.remove_oomc(cap_df, 'iso 3 code')
    cap_df.rename(columns={'iso 3 code': 'tile_ID'}, inplace=True)
    return cap_df


def get_tile_coord(cap_df, label):
    lbl_lat = cap_df.loc[cap_df['tile_ID'] == label, 'lat'].item()
    lbl_lng = cap_df.loc[cap_df['tile_ID'] == label, 'lng'].item()
    return lbl_lat, lbl_lng


def haversine_distance(lat1, lon1, lat2, lon2):
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def add_flow_distances(df, cap_df, label):
    df = df.transpose()
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'tile_ID'}, inplace=True)
    df = df.merge(cap_df, how="inner", on="tile_ID")
    lat_1, lng_1 = get_tile_coord(cap_df, label)

    flow_dist_list = []
    for i in range(len(df)):
        lat_2 = df.loc[i, 'lng']
        lng_2 = df.loc[i, 'lng']
        flow_dist_list.append(haversine_distance(lat_1, lng_1, lat_2, lng_2))

    flow_list_95 = df[0] * flow_dist_list
    flow_list_00 = df[1] * flow_dist_list
    flow_list_05 = df[2] * flow_dist_list
    flow_list_10 = df[3] * flow_dist_list

    df[1995] = flow_list_95
    df[2000] = flow_list_00
    df[2005] = flow_list_05
    df[2010] = flow_list_10
    return df


def get_top_flow_distances(df):
    df.drop([0, 1, 2, 3, 'country', 'world_region', 'capital', 'lng', 'lat'], axis=1, inplace=True)
    df['Total'] = df.sum(axis=1)
    df.sort_values(['Total'], ascending=False, axis=0, inplace=True)
    df_top10 = df.head(10)
    df_top10.drop('Total', axis=1, inplace=True)
    df_top10 = df_top10.transpose()
    df_top10 = df_top10.rename(columns=df_top10.iloc[0]).drop(df_top10.index[0])
    df_top10.index = df_top10.index.map(int)

    return df_top10
