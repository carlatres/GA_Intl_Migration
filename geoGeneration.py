import logging
# import pandas as pd
import numpy as np
from countryinfo import CountryInfo


def generate_latlon(df):
    longitude = []
    latitude = []
    for index, row in df.iterrows():
        try:
            x = CountryInfo(row['country'])
            geo = x.capital_latlng()
            latitude.append(geo[0])
            longitude.append(geo[1])
        except Exception as e:
            logging.info(e)
            latitude.append(np.nan)
            longitude.append(np.nan)
    df["lng"] = longitude
    df["lat"] = latitude
    return df
