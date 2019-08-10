import pandas as pd
from sklearn.model_selection import train_test_split

BUSINESS_TRAIN = '/Users/jaewooklee/farmers_market/twitter/data/bus_nobus_cleaned.xlsx'
LABELING_TRAIN = '/Users/jaewooklee/farmers_market/twitter/data/twitter_1000_labelling_cleaned.xlsx'
LABEL_RAW = '/Users/jaewooklee/farmers_market/twitter/data/twitter_1000_labelling_cleaned_new.xlsx'
YELP_TRAIN = '/Users/jaewooklee/farmers_market/yelp/data/2000_yelp_labeled.csv'
def parse_excel_bus_nobus():
    df = pd.read_excel(BUSINESS_TRAIN)
    df_bus = df[df.Label == 'Business']
    df_nobus = df[df.Label != 'Business']

    return (df_bus, df_nobus)

def parse_excel_labels():
    df = pd.read_excel(LABEL_RAW)

    # df=remove_white_space(df)
    df_aval = df[df.Label == 'availability']
    df_environ = df[df.Label =='environment']
    df_quality = df[df.Label =='quality']
    df_safety = df[df.Label =='safety']
    df_nonrel = df[df.Label == "NR"]

    return (df_aval, df_environ, df_quality, df_safety, df_nonrel)

def parse_excel_NR():
    df = pd.read_excel(LABELING_TRAIN)
    df.dropna()
    df_NR = df[df.Label == 'NR']
    df_R = df[df.Label != 'NR']
    df_R = df_R.replace({'environment': 'R', 'availability': 'R', 'quality': 'R', 'safety': 'R'})

    return (df_NR, df_R)

def remove_white_space(dataframe):
    for index, row in dataframe.iterrows():
        try:
            row['Label'] = row['Label'].rstrip().lstrip()
        except:
            print(index)
            print(row['Label'])
    return dataframe

def parse_yelp():
    df = pd.read_csv(YELP_TRAIN)

    df=remove_white_space(df)
    df_aval = df[df.Label == 'availability']
    df_environ = df[df.Label =='environment']
    df_quality = df[df.Label =='quality']
    df_safety = df[df.Label =='safety']
    df_nonrel = df[df.Label == "non-relevant"]

    df_nonrel = df_nonrel.replace({"non-relevant": 'NR'})


    return (df_aval, df_environ, df_quality, df_safety, df_nonrel)
