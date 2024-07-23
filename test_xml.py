import pandas as pd
import xml.etree.ElementTree as et

# THIS IS THE CODE TO ADD THE CORRESPONDING GDP

xtree = et.parse('data/gdp.xml')
xroot = xtree.getroot()

df_cols = [
    "Country or Area",
    "SNA93 Table Code",
    "Sub Group",
    "Item",
    "SNA93 Item Code",
    "Year",
    "Series",
    "Currency",
    "SNA System",
    "Fiscal Year Type",
    "Value",
    "Value Footnotes"
]
rows = []

df = pd.DataFrame(columns=df_cols)

for i in xroot.iter(tag='record'):
    aux = []
    for j in i:
        aux.append(j.text)
    df = df.append(pd.Series(aux, index=df_cols), ignore_index=True)

df = df.drop_duplicates('Country or Area')
df = df[["Country or Area", "Currency", "Value"]]
eur = []

currency = {
    'ALGERIANDINAR': 0.0073,
    '(SECOND)KWANZA': 0.002,
    'USDOLLAR': 1,
    'USDOLLARS': 1,
    'EURO': 0.94,
    'ECDOLLAR': 0.37,
    'ARGENTINEPESO': 0.26,
    'DRAM': 0.0025,
    'AUSTRALIANDOLLAR': 0.93,
    'AUSTRALIANDOLLARS': 0.93,
    'AZERBAIJANNEWMANAT': 0.59,
    'BAHAMIANDOLLAR': 1,
    'BAHRAINDINAR': 2.65,
    'TAKA': 0.014,
    'BELARUSSIANROUBLE(RE-DENOM1:1000)': 0.39,
    'BELIZEDOLLAR': 0.5,
    'CFAFRANC': 0.0016,
    'BERMUDADOLLAR': 1,
    'NGULTRUM': 0.012,
    'BOLIVIANO': 0.14,
    'PULA': 0.076,
    'REAL': 0.18,
    'BRUNEIDOLLAR': 0.75,
    'LEV': 0.03,
    'ESCUDO': 0.0097,
    'BELARUS': 0.4,
    'CANADIANDOLLAR': 0.75,
    'CHILEANPESO': 0.0020,
    'YUANRENMINBI': 0.15,
    'DANISHKRONE': 0.14,
    'EGYPTIANPOUND': 0.033,
    'LILANGENI': 0.056,
    'ETHIOPIANBIRR': 0.019,
    'HONGKONGDOLLAR': 0.13,
    'COLOMBIANPESO': 0.00051,
    'COSTARICANCOLON': 0.0018,
    'KUNA': 0.1419,
    'CUBANPESO': 0.042,
    'ICELANDICKRÓNA': 0.00706,
    'INDIANRUPEE': 0.22,
    'IRANIANRIAL': 0.00009,
    'NETHERLANDSANTILLEANGUILDER': 0.55,
    'FORINT': 0.0055,
    'INDONESIANRUPIAH': 0.00011,
    'CZECHKORUNA': 0.58,
    'GUINEANFRANC': 0.0002,
    'GUYANADOLLAR': 0.0047,
    'LEMPIRA': 0.054,
    'NEWSHEQEL': 0.26,
    'JAMAICANDOLLAR': 0.012,
    'CONGOLESEFRANC': 0.0011,
    'FIJIDOLLAR': 0.53,
    'LARI': 0.43,
    'QUETZAL': 0.12,
    'IRAQIDINAR': 0.00088,
    'YEN': 0.013,
    'JORDANDINAR': 1.42,
    "PA'ANGA": 0.42,
    'TUNISIANDINAR': 0.32,
    'NEWTURKISHLIRA': 0.53,
    'TENGE': 0.0022,
    'SAMONI': 0.16,
    'TANZANIANSHILLING': 0.00076,
    'BAHT': 0.033,
    'OUGUIYA': 0.037,
    'RAND': 0.14,
    'SOUTHSUDANESEPOUND': 0.0077,
    'SRILANKARUPEE': 0.0089,
    'SDG': 0.37,
    'KENYASHILLINGS': 0.014,
    'LEONE': 0.00023,
    'SINGAPOREDOLLAR': 0.71,
    'SURINAMEDOLLAR': 0.15,
    'LIBYANDINAR': 0.82,
    'DINAR': 0.00068,
    'SEYCHELLESRUPEE': 0.071,
    'SWEDISHKRONA': 0.096,
    'SYRIANPOUND': 0.000398,
    'SWISSFRANC': 1.23,
    'RUSSIANRUBLE(RE-DENOM1:1000)': 0.034,
    'SAUDIARABIANRIYAL': 0.27,
    'RINGGIT': 0.31,
    'KOREANWON': 0.00079,
    'MOLDOVANLEU': 0.082,
    'NEWROMANIANLEU': 0.22,
    'KUWAITIDINAR': 3.51,
    'NAIRA': 0.0065,
    'TFYRMACEDONIANDENAR': 0.0174,
    'PAKISTANRUPEE': 0.0037,
    'QATARRIYAL': 0.2747,
    'KYRGYZSOM': 0.02,
    'NEPALESERUPEE': 0.0076,
    'CÓRDOBA': 0.027,
    'NORWEGIANKRONE': 0.098,
    'PHILIPPINEPESO': 0.018,
    'LEBANESEPOUND': 0.0006,
    'NAMIBIADOLLAR': 0.067,
    'NEWZEALANDDOLLAR': 0.63,
    'BALBOA': 1,
    'LOTI': 0.13,
    'MOROCCANDIRHAM': 0.098,
    'METICAL': 0.032,
    'KYAT': 0.00048,
    'RIALOMANI': 2.6,
    'GUARANI': 0.0002,
    'ZLOTY': 0.35,
    'RUFIYAA': 0.079,
    'MAURITIANRUPEE': 0.0219,
    'MEXICANNEWPESO': 0.076,
    'TOGROG': 0.0003,
    'NEWSOL': 0.000001,
    'UZBEKSUM': 0.0000883316,
    'VATU': 0.0085,
    'BOLÍVARSOBERANO': 0.041,
    'HRYVNIA': 0.027,
    'POUNDSTERLING': 1.21,
    'URUGUAYANPESO': 0.053,
    'UAEDIRHAM': 0.27,
    'YEMENIRIAL': 0.0040,
    'ZAMBIANKWACHA': 0.19
}

for i in df.iterrows():
    try:
        xxx = float(i[1]['Value']) * currency[i[1]['Currency'].upper().replace(' ', '').replace('.', '')]
    except Exception as e:
        xxx = None
    eur.append(xxx)
df['usd_value'] = eur

df2 = pd.read_csv('data/countries.csv')
df_merge = df.merge(df2, left_on='Country or Area', right_on='country', how='left')
df_merge = df_merge[['tile_ID', 'country', 'usd_value', 'Country or Area']]
df_merge.to_csv('./data/out_put.csv')
z = 0
