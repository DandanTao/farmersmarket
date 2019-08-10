#!/usr/local/bin/python3

import pandas as pd
import os
import numpy as np
import json
import csv
import re #regular expression
import string
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import preprocessor_v0

BUS_VS_NONBUS_CSV = '/Users/jaewooklee/farmers_market/twitter/data/bus_nobus.csv'
LABELING_CSV = '/Users/jaewooklee/farmers_market/twitter/data/twitter_FM_1000_labelling.csv'
LABEL = '/Users/jaewooklee/farmers_market/twitter/data/twitter_FM_1000_labelling.xlsx'
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"
         u"\U0001F300-\U0001F5FF"
         u"\U0001F680-\U0001F6FF"
         u"\U0001F1E0-\U0001F1FF"
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

emoticons = emoticons_happy.union(emoticons_sad)

REL_LABEL = set(['environment' 'NR' 'availability' 'quality' 'safety'])

def remove_punctuation(sen):
    regex = re.compile("[{}]".format(re.escape(string.punctuation)))
    return regex.sub("", sen)

def replace_digit(sen):
    return re.sub('\d', '', sen)

def filter_word(sen):
    token = sen.split(' ')
    new_tok = []
    for tok in token:
        if len(tok) > 2:
            new_tok.append(tok)
    return ' '.join(new_tok)

def clean_dataframe(file):
	df = pd.read_csv(file)
	try:
		df.drop(columns=['No.', 'Unnamed: 0', 'Words', 'Notes', 'Contributor'], inplace=True)
	except:
		df.drop(columns=['No.', 'Words', 'Notes', 'Contributor'], inplace=True)

	# df = df.replace('education', 'NR')
	# df = df.replace('behavior', 'NR')
	# df = df.replace('convenience', 'NR')
	# df = df.replace('politics', 'NR')
	# df = df.replace('economics', 'NR')
	return df.dropna()

def clean_dataframe_v1(file):
	df = pd.read_excel(file)
	# try:
	# 	df.drop(columns=['No.', 'Unnamed: 0', 'Words', 'Notes', 'Contributor'], inplace=True)
	# except:
	# 	df.drop(columns=['No.', 'Words', 'Notes', 'Contributor'], inplace=True)
    #
	# df = df.replace('education', 'NR')
	# df = df.replace('behavior', 'NR')
	# df = df.replace('convenience', 'NR')
	# df = df.replace('politics', 'NR')
	# df = df.replace('economics', 'NR')
	return df.dropna()

def clean_tweets(tweet):
    stop_words = set(stopwords.words('english'))
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)

    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)

    tweet = emoji_pattern.sub(r'', tweet)
    word_tokens = word_tokenize(tweet)

    filtered_tweet = []

    for w in word_tokens:
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)

    return ' '.join(filtered_tweet)

def clean_all_tweets_in_df(df, nop=False):
    for i, content in enumerate(df.Tweet):
        if nop:
            df.at[i, 'Tweet'] = clean_tweets(content)
        else:
            cleaned = clean_tweets(p.clean(content).lower())
            # cleaned = preprocessor_v0.twitter_preprocess(cleaned)
            cleaned = remove_punctuation(cleaned)
            cleaned = replace_digit(cleaned)
            cleaned = filter_word(cleaned)
            df.at[i, 'Tweet'] = cleaned

    df.rename(columns={'Tweet':'Sentences'}, inplace=True)

def process_df(csv_file, filename, nop):
    df = clean_dataframe(csv_file)
    clean_all_tweets_in_df(df, nop)
    path = os.path.join('/'.join(csv_file.split('/')[:-1]), filename)
    df.to_excel(path, index=False)

def process_df_excel(excel_file, filename, nop):
    df = clean_dataframe_v1(excel_file)
    clean_all_tweets_in_df(df, nop)
    path = os.path.join('/'.join(excel_file.split('/')[:-1]), filename)
    df.to_excel(path, index=False)

def process_all():
    # process_df(BUS_VS_NONBUS_CSV, 'bus_nobus_cleaned_nop.xlsx', nop=True)
    process_df(BUS_VS_NONBUS_CSV, 'bus_nobus_cleaned.xlsx', nop=False)
    # process_df(LABELING_CSV, 'twitter_1000_labelling_cleaned_nop.xlsx', nop=True)
    process_df(LABELING_CSV, 'twitter_1000_labelling_cleaned.xlsx', nop=False)
    process_df_excel(LABEL, 'twitter_1000_labelling_cleaned_new.xlsx', nop=False)
