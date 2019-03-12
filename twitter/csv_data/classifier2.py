import pandas as pd
df = pd.read_csv('twitter_FM_1000_labelling.csv');

removed_rows = [];
num_env = 0;
for index, row in df.iterrows():
    if ((row['Notes'] == 'Business') or (row['Notes'] == 'business (content)') or (row['Notes'] == 'business (time, location)')):
        df.loc[index, 'Label'] = 'Business'
        #print(df.iloc[i]['Label']);
    if row['Label'] == 'environment':
        num_env = num_env + 1;
        if num_env > 100:
            removed_rows.append(index)

for index, row in df.iterrows():
    if (row['Label'] == 'NR' or row['Label'] == 'Business'):
        removed_rows.append(index);

df.drop(df.index[removed_rows], inplace=True)

df.to_csv('new_tweet_classifier2.csv');
