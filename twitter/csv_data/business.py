import pandas as pd
df = pd.read_csv('twitter_FM_1000_labelling.csv');

for index, row in df.iterrows():
    if ((row['Notes'] == 'Business') or (row['Notes'] == 'business (content)') or (row['Notes'] == 'business (time, location)')):
        df.loc[index, 'Label'] = 'Business'
        #print(df.iloc[i]['Label']);
removed_rows = [];
for index, row in df.iterrows():
    if (row['Label'] == 'NR'):
        removed_rows.append(index);

df.drop(df.index[removed_rows], inplace=True)

df.to_csv('new_tweet.csv');
