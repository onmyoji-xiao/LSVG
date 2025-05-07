import argparse
import pandas as pd
from ast import literal_eval
from dataset.utils import *


def get_referit3d_classes(csv_path):
    train_data = pd.read_csv(csv_path)
    train_data = train_data[['tokens', 'stimulus_id']]
    train_data.tokens = train_data['tokens'].apply(literal_eval)
    with open('./mappings/scannet_instance_class_to_semantic_class.json') as fin:
        ins_sem_mapping = json.load(fin)
    words = set()
    for ori_tokens in train_data['tokens']:
        related_words = spacy_isin_text(set(list(ins_sem_mapping.keys())), ori_tokens)
        for classname in related_words:
            words.add(classname)
    for i, stm in enumerate(train_data['stimulus_id']):
        _, instance_label, _, _, _ = decode_stimulus_string(stm)
        words.add(instance_label)
        print(i)
    for w in words:
        if w in ins_sem_mapping and 'other' in ins_sem_mapping[w]:
            ins_sem_mapping[w] = w
            print(w)
    for key in ins_sem_mapping:
        if 'other' in ins_sem_mapping[key]:
            ins_sem_mapping[key] = 'other'
    with open('./mappings/nr3d_classes_mapping2.json', 'w') as f:
        json.dump(ins_sem_mapping, f, indent=4)


def get_scanrefer_classes(train_file):
    with open(train_file) as fs:
        train_js = json.load(fs)  # maxlen:126
    with open('./mappings/scannet_instance_class_to_semantic_class.json') as fin:
        ins_sem_mapping = json.load(fin)
    words = set()
    for data in train_js:
        instance_label = data['object_name']
        words.add(instance_label)
    for w in words:
        if w in ins_sem_mapping and 'other' in ins_sem_mapping[w]:
            ins_sem_mapping[w] = w
            print(w)
    for key in ins_sem_mapping:
        if 'other' in ins_sem_mapping[key]:
            ins_sem_mapping[key] = 'other'
    with open('./mappings/scanrefer_classes_mapping.json', 'w') as f:
        json.dump(ins_sem_mapping, f, indent=4)

if __name__ == '__main__':
    get_referit3d_classes('./referit3d/nr3d_train.csv')
    # get_scanrefer_classes('./scanrefer/ScanRefer_filtered_train.json')