import pandas as pd
from sklearn.model_selection import train_test_split

def remove_white_space(dataframe):
    for index, row in dataframe.iterrows():
        try:
            row['Label'] = row['Label'].rstrip().lstrip()
        except:
            pass
    return dataframe

def parse_csv_by_class_two_file(file1, file2):
    df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class_v0(file1)
    df_aval1, df_environ1, df_quality1, df_safety1, df_nonrel1 = parse_csv_by_class_v1(file2)
    return (pd.concat([df_aval, df_aval1], sort=False),
            pd.concat([df_environ, df_environ1], sort=False),
            pd.concat([df_quality, df_quality1], sort=False),
            pd.concat([df_safety, df_safety1], sort=False),
            pd.concat([df_nonrel, df_nonrel1], sort=False))

def parse_csv_by_class_v1(file):
    df = pd.read_csv(file)

    df=remove_white_space(df)
    df_aval = df[df.Label == 'availability']
    df_environ = df[df.Label =='environment']
    df_quality = df[df.Label =='quality']
    df_safety = df[df.Label =='safety']
    df_nonrel = df[df.Label == "Non-relevant"]

    return (df_aval, df_environ, df_quality, df_safety, df_nonrel)

def parse_csv_by_class_v0(file):
    df = pd.read_csv(file)
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']
    df=remove_white_space(df)
    df=df.fillna("Non-relevant")

    df_aval = df[df.Label.str.contains('availability')]
    df_aval[:]['Label'] = 'availability'
    df_environ = df[df.Label.str.contains('environment')]
    df_environ[:]['Label'] = 'environment'
    df_quality = df[df.Label.str.contains('quality')]
    df_quality[:]['Label'] = 'quality'
    df_safety = df[df.Label.str.contains('safety')]
    df_safety[:]['Label'] = 'safety'
    df_nonrel = df[df.Label == "Non-relevant"]

    return (df_aval, df_environ, df_quality, df_safety, df_nonrel)

def parse_csv_relevant_non_relevant(file):
    """
    reads csv file data with labels and comment
    Parse into 2 category: Relevant and Non-relevant
    """
    df = pd.read_csv(file)

    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("Non-relevant")

    for index, row in df.iterrows():
        if row['Label'] != 'Non-relevant':
            row['Label'] = 'relevant'

    return train_test_split(df, test_size=0.2)

def parse_csv_remove_multiclass(file):
    """
    reads csv file data with labels and comment
    Discards only multiclass objects
    """
    df = pd.read_csv(file)

    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("None")
    df = remove_white_space(df)
    # drop all sentence with multiple labels
    df=df[~df.Label.str.contains(",")]

    # drop all sentences with label 'covinience'
    # df = df.replace('covinience', 'convenience')
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']

    return train_test_split(df, test_size=0.2)

def parse_csv_discard_non_relevant(file):
    """
    reads csv file data with labels and comment
    Discards all non-relevant sentences and multiclass sentences
    """
    import pandas as pd
    df = pd.read_csv(file)

    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("None")
    df=remove_white_space(df)
    # drop all sentence with multiple labels
    df=df[df.Label != 'None']
    df=df[~df.Label.str.contains(",")]

    # drop all sentences with label 'covinience'
    # df = df.replace('covinience', 'convenience')
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']

    return train_test_split(df, test_size=0.2)

def random_test_data(file, size=0.2):
    df = pd.read_csv(file)
    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("Non-relevant")
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']
    df=remove_white_space(df)
    _, test = train_test_split(df, test_size=size)
    return test

def random_test_data_v1(file, size=0.2):
    df = pd.read_csv(file)
    df=remove_white_space(df)
    _, test = train_test_split(df, test_size=size)
    return test
