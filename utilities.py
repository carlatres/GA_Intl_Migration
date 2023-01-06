import csv
import pandas as pd


def to_csvfile(file_name, d):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(d)


# remove countries out of the geopandas low-res map (out of map countries)
def remove_oomc(df, col):
    df = df.drop(df[(df[col] == 'ABW') | (df[col] == 'BHR') | (df[col] == 'BRB') | (df[col] == 'CHI') |
                    (df[col] == 'COM') | (df[col] == 'CPV') | (df[col] == 'FSM') | (df[col] == 'GLP') |
                    (df[col] == 'GRD') | (df[col] == 'GUF') | (df[col] == 'GUM') | (df[col] == 'HKG') |
                    (df[col] == 'LCA') | (df[col] == 'MAC') | (df[col] == 'MDV') | (df[col] == 'MLT') |
                    (df[col] == 'MTQ') | (df[col] == 'MUS') | (df[col] == 'MYT') | (df[col] == 'PYF') |
                    (df[col] == 'REU') | (df[col] == 'SGP') | (df[col] == 'STP') | (df[col] == 'TON') |
                    (df[col] == 'VCT') | (df[col] == 'VIR') | (df[col] == 'WSM')].index)
    return df


# remove countries out of the geopandas low-res map in the origin and destination columns
def remove_orig_dest_countries(df):
    df = remove_oomc(df, 'origin')
    df = remove_oomc(df, 'destination')
    return df
