import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
import os
import pathlib
import json


def trim_scans_per_referit3d_data(referit_data, scans):
    # remove scans not in referit_data
    in_r3d = referit_data.scan_id.unique()
    to_drop = []
    for k in scans:
        if k not in in_r3d:
            to_drop.append(k)
    for k in to_drop:
        del scans[k]
    print('Dropped {} scans to reduce mem-foot-print.'.format(len(to_drop)))
    return scans


def mean_color(scan_ids, all_scans):
    mean_rgb = np.zeros((1, 3), dtype=np.float32)
    n_points = 0
    for scan_id in scan_ids:
        if isinstance(all_scans[scan_id],list):
            color = all_scans[scan_id][0].color
        else:
            color = all_scans[scan_id].color
        mean_rgb += np.sum(color, axis=0)
        n_points += len(color)
    mean_rgb /= n_points  #
    return mean_rgb


def load_filtered_data(pkl_scannet_file, refers, use_sem, ins_sem_mapping, add_pad=True):
    pkls = pkl_scannet_file.split(';')
    all_scans = dict()
    for pkl_f in pkls:
        with open(pkl_f, 'rb') as f:
            scans = pickle.load(f)
        scans = {scan.scan_id: scan for scan in scans}
        all_scans.update(scans)

    class_labels = set()
    for k, scan in all_scans.items():
        idx = np.array([o.object_id for o in scan.three_d_objects])
        if use_sem:
            class_labels.update([ins_sem_mapping[o.instance_label] for o in scan.three_d_objects if
                                 o.instance_label in ins_sem_mapping])
        else:
            class_labels.update([o.instance_label for o in scan.three_d_objects])
        assert np.all(idx == np.arange(len(idx)))

    referit_data = pd.concat([pd.read_csv(rf) for rf in refers])

    class_to_idx = {}
    i = 0
    for el in sorted(class_labels):
        class_to_idx[el] = i
        i += 1

    pad_i = len(class_to_idx)
    if add_pad:
        class_to_idx['pad'] = pad_i

    referit_data = referit_data[['tokens', 'instance_type', 'scan_id', 'is_train',
                                 'dataset', 'target_id', 'utterance', 'stimulus_id']]
    referit_data.tokens = referit_data['tokens'].apply(literal_eval)

    return all_scans, class_to_idx, referit_data


def load_scanrefer_data(pkl_scannet_file, train_file, val_file, use_sem, ins_sem_mapping, add_pad=True):
    pkls = pkl_scannet_file.split(';')
    all_scans = dict()
    for pkl_f in pkls:
        with open(pkl_f, 'rb') as f:
            scans = pickle.load(f)
        for scan in scans:
            if scan.scan_id not in all_scans:
                all_scans[scan.scan_id] = [scan]
            else:
                all_scans[scan.scan_id].append(scan)

    class_labels = set()
    for k, scan_list in all_scans.items():
        for scan in scan_list:
            if use_sem:
                class_labels.update([ins_sem_mapping[o.instance_label] for o in scan.three_d_objects])
            else:
                class_labels.update([o.instance_label for o in scan.three_d_objects])

    class_to_idx = {}
    i = 0
    for el in sorted(class_labels):
        class_to_idx[el] = i
        i += 1

    # Add the pad class needed for object classification
    if add_pad:
        class_to_idx['pad'] = len(class_to_idx)

    with open(train_file) as fs:
        train_js = json.load(fs)  # maxlen:126
    with open(val_file) as fs:
        val_js = json.load(fs)  # maxlen:108

    train_ids = sorted(list(set([data["scene_id"] for data in train_js])))  # 562
    val_ids = sorted(list(set([data["scene_id"] for data in val_js])))  # 141

    scanrefer = dict()
    scanrefer['train'] = train_js
    scanrefer['val'] = val_js

    scans_split = dict()
    scans_split['train'] = set(train_ids)
    scans_split['val'] = set(val_ids)

    return all_scans, scans_split, scanrefer, class_to_idx
