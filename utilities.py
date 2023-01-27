import csv
from math import radians, cos, sin, sqrt, atan2


# pass dataframe to csv file
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


# calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    r = 6371  # radius of Earth in kilometers
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = r * c
    return distance


# calculate the distance between two points using the Haversine distance
def calculate_distance(flows_df, capital_df):

    print('ciao')


