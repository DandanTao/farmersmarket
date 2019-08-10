from merge_sentences import BASE_PATH
import os
import pandas as pd

BUSINESS = os.path.join(BASE_PATH, '../data/bus_nobus_final.xlsx')
TWITTER_LABELS =  os.path.join(BASE_PATH, '../data/twitter_label_final.xlsx')

twitter_label = set(['NR', 'availability', 'environment', 'quality', 'safety'])

def append_to_dict(txt, label, dict):
    for sen in txt.split('\n'):
        if len(sen) > 0:
            dict['Sentences'].append(sen)
            dict['Label'].append(label)

def parse_file_and_append(file, label, dict):
    with open(file) as f:
        append_to_dict(f.read(), label, dict)

def merge_to_df():
    twitter_files = {'Sentences':[], 'Label':[]}
    business_files = {'Sentences':[], 'Label':[]}

    for file in os.listdir(BASE_PATH):
        if not file.startswith('label'):
            continue
        label = file.split('.txt')[0].split('-')[1]
        filepath = os.path.join(BASE_PATH, file)
        if label in twitter_label:
            parse_file_and_append(filepath, label, twitter_files)
        else:
            parse_file_and_append(filepath, label, business_files)

    twitter_df = pd.DataFrame(twitter_files)
    business_df = pd.DataFrame(business_files)

    twitter_df.to_excel(TWITTER_LABELS, index=False)
    business_df.to_excel(BUSINESS, index=False)
