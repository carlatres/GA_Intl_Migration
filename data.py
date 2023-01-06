import pandas as pd
# import numpy as np
from countryinfo import CountryInfo
# from geopy.exc import GeocoderTimedOut
# from geopy.geocoders import Nominatim
import utilities as utils


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
    # print(country_list)
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
