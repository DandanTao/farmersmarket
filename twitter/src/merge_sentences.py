import pandas as pd
import numpy as np
import os

NOP_BUS_NOBUS = '/Users/jaewooklee/farmers_market/twitter/data/bus_nobus_cleaned_nop.xlsx'
BUS_NOBUS = '/Users/jaewooklee/farmers_market/twitter/data/bus_nobus_cleaned.xlsx'
NOP_LABEL = '/Users/jaewooklee/farmers_market/twitter/data/twitter_1000_labelling_cleaned_nop.xlsx'
LABEL = '/Users/jaewooklee/farmers_market/twitter/data/twitter_1000_labelling_cleaned.xlsx'
BASE_PATH = '/Users/jaewooklee/farmers_market/twitter/separate_sentences/'

def push_to_dict(df, sentence_dict):
    for i, row in df.iterrows():
        label = row.Label.replace(' ', '_')
        if label not in sentence_dict:
            sentence_dict[label] = []
        if len(row.Sentences) > 3:
            sentence_dict[label].append(row.Sentences)

    return sentence_dict

def merge_sentence_to_split(filename, dict):
    for key, value in dict.items():
        all_sen = '.\n'.join(value)
        f_name = BASE_PATH + filename.split('.')[0]+'-'+key+'.txt'
        with open(f_name, 'w') as f:
            f.write(all_sen)

def merge_all():
    # nop_bus_nobus_df = pd.read_excel(NOP_BUS_NOBUS)
    bus_nobus_df = pd.read_excel(BUS_NOBUS)
    # nop_label_df = pd.read_excel(NOP_LABEL)
    label_df = pd.read_excel(LABEL)

    # label_bus_nop = {}
    label_bus = {}
    # label_nop = {}
    label = {}

    # push_to_dict(nop_bus_nobus_df, label_bus_nop)
    push_to_dict(bus_nobus_df, label_bus)
    # push_to_dict(nop_label_df, label_nop)
    push_to_dict(label_df, label)

    # merge_sentence_to_split("label_bus_nop", label_bus_nop)
    merge_sentence_to_split("label_bus", label_bus)
    # merge_sentence_to_split("label_nop", label_nop)
    merge_sentence_to_split("label", label)
