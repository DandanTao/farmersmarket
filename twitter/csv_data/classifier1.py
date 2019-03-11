import pandas as pd
df = pd.read_csv('twitter_FM_1000_labelling.xlsx');

for index, row in df.iterrows():
    if ((row['Notes'] == 'Business') or (row['Notes'] == 'business (content)') or (row['Notes'] == 'business (time, location)')):
        df.loc[index, 'Label'] = 'Business'
    else:
        df.loc[index, 'Label'] = 'Not Business'
#df.drop(df.index[removed_rows], inplace=True)

df.to_csv('new_tweet_classifier1.csv');
