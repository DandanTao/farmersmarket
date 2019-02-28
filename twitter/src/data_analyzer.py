import json
import pandas as pd
import csv
import sys
import re
import csv
from bs4 import BeautifulSoup
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

RUN_PATH = '../'
tweets_csv_name = 'twitter_data_2132019'
tweets_data_path = 'JSON_data/'+tweets_csv_name

tweets_data = []

def import_json_tweets():
    tweets_file = open(tweets_data_path+'.json', "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue
    return

def text_cleanup(tweet_text):
    tweet_text = BeautifulSoup(tweet_text, 'lxml').get_text();
    tweet_text = re.sub('https?://[A-Za-z0-9./]+','', tweet_text);
    tweet_text = tweet_text.replace("https:/", "");
    tweet_text = ''.join([c for c in tweet_text if ord(c) < 128]);
    tweet_text = re.sub('^RT @[A-Za-z0-9_]+: ', '', tweet_text);
    tweet_text = re.sub('  ', ' ', tweet_text);
    return tweet_text

def generate_dataframe():
    columns = ['time', 'text', 'location']
    df = pd.DataFrame(columns = columns);
    for i in range(0, len(tweets_data)-1):
        if i%100 == 0:
            print(100 * i/len(tweets_data), end = '\r');
        if isinstance(tweets_data[i], int) == False:
            if tweets_data[i]['place']!=None:
                tweets_place = tweets_data[i]['place']['name']
            else:
                tweets_place = "None"
            if 'extended_tweet' in tweets_data[i]:
                 tweet_text = tweets_data[i]['extended_tweet']['full_text'];
            elif 'retweeted_status' in tweets_data[i]:
                if 'extended_tweet' in tweets_data[i]['retweeted_status']:
                    tweet_text = tweets_data[i]['retweeted_status']['extended_tweet']['full_text'];
                else: tweet_text = tweets_data[i]['text']
            else: tweet_text = tweets_data[i]['text']

            tweet_text = text_cleanup(tweet_text)

            df.loc[i] = [tweets_data[i]['created_at'], tweet_text, tweets_place]

    df.to_csv("csv_data/"+tweets_csv_name+".csv")
    return

def main():
    os.chdir(RUN_PATH)

    import_json_tweets()
    print("Total Number of Tweets Imported = ", len(tweets_data))
    generate_dataframe()

if __name__ == '__main__':
    main()
