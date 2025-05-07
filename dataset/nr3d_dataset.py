import logging
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from functools import partial
from dataset.utils import *
from dataset.transform import mean_rgb_unit_norm_transform
from torch.utils.data import DataLoader
import h5py
import open_clip


class Nr3dDataset(Dataset):
    def __init__(self, references, scans, points_per_object, max_distractors, use_sem, ins_sem_mapping, clip_info,
                 clip_dim, class_to_idx=None, object_transformation=None, visualization=False, train=False):

        self.references = references
        self.scans = scans
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        self.ins_sem_mapping = ins_sem_mapping
        self.train = train
        self.use_sem = use_sem
        self.class_num = len(set(self.class_to_idx.values())) - 1
        self.clip_path = clip_info.split(';')
        self.clip_dim = clip_dim

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan_id = ref['scan_id']
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        ori_tokens = ref['tokens']
        tokens = " ".join(ori_tokens)

        related_words = spacy_isin_text(set(list(self.ins_sem_mapping.keys()) + list(self.ins_sem_mapping.values())),
                                        ori_tokens)
        related_class = set()
        if self.use_sem:
            for classname in related_words:
                related_class.add(self.ins_sem_mapping[classname] if classname in self.ins_sem_mapping else classname)
        else:
            related_class = related_words
        return scan, target, tokens, scan_id, related_class, related_words

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label
        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def sem_prepare_distractors(self, scan, target):
        target_label = self.ins_sem_mapping[target.instance_label]
        distractors = [o for o in scan.three_d_objects if
                       (self.ins_sem_mapping[o.instance_label] == target_label and (o != target))]
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if self.ins_sem_mapping[o.instance_label] not in already_included]

        np.random.shuffle(clutter)
        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def __getitem__(self, index):
        res = dict()

        scan, target, tokens, scan_id, related_class, related_words = self.get_reference_data(index)

        # Make a context of distractors
        if self.use_sem:
            context = self.sem_prepare_distractors(scan, target)
        else:
            context = self.prepare_distractors(scan, target)
        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        if len(related_class) > 0:
            res['related_class'] = ','.join(related_class)
        else:
            res['related_class'] = ''

        object_dict = {}
        if '_00' in scan_id:
            h5path = self.clip_path[0]
        else:
            h5path = self.clip_path[1]
        with h5py.File(h5path, "r") as f:
            for d_id in f[scan_id]:
                h5_feats = np.array(f[scan_id][d_id])
                if self.train:
                    ri = random.randint(0, h5_feats.shape[0] - 1)
                else:
                    ri = 0
                object_dict[int(d_id)] = h5_feats[ri]
        clip_feats = []
        for i, o in enumerate(context):
            if o.object_id in object_dict:
                clip_feats.append(object_dict[o.object_id])
            else:
                clip_feats.append(np.zeros((self.clip_dim,)))

        clip_feats = np.stack(clip_feats, axis=0)
        if len(clip_feats) < self.max_context_size:
            clip_feats = np.concatenate(
                [clip_feats, np.zeros((self.max_context_size - len(clip_feats), self.clip_dim))], axis=0)
        res['clip_feats'] = clip_feats

        samples = np.array([sample_scan_object(o, scan, self.points_per_object) for o in context])

        # mark their classes
        box_info = np.zeros((self.max_context_size, 6))  # (52,6)
        box_info[:len(context), 0] = [o.get_bbox().cx for o in context]
        box_info[:len(context), 1] = [o.get_bbox().cy for o in context]
        box_info[:len(context), 2] = [o.get_bbox().cz for o in context]
        box_info[:len(context), 3] = [o.get_bbox().lx for o in context]
        box_info[:len(context), 4] = [o.get_bbox().ly for o in context]
        box_info[:len(context), 5] = [o.get_bbox().lz for o in context]
        box_corners = np.zeros((self.max_context_size, 8, 3))
        box_corners[:len(context)] = [o.get_bbox().corners for o in context]
        if self.object_transformation is not None:
            samples = self.object_transformation(samples)

        res['scan_id'] = scan_id
        res['context_size'] = len(samples)
        res['objects'] = pad_samples(samples, self.max_context_size)

        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool_)
        if self.use_sem:
            res['class_labels'] = semantic_labels_of_context(context, self.max_context_size, self.class_to_idx,
                                                             self.ins_sem_mapping)
            target_class_mask[:len(context)] = [
                self.ins_sem_mapping[target.instance_label] == self.ins_sem_mapping[o.instance_label] for o in context]
            res['target_class'] = self.class_to_idx[self.ins_sem_mapping[target.instance_label]]
        else:
            res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)
            target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]
            res['target_class'] = self.class_to_idx[target.instance_label]

        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['box_info'] = box_info
        res['box_corners'] = box_corners

        object_ids = np.zeros((self.max_context_size))
        for k, o in enumerate(context):
            object_ids[k] = o.object_id
        res['object_ids'] = object_ids
        if self.visualization:
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['target_object_id'] = target.object_id

        return res


def make_data_loaders(cfg, referit_data, class_to_idx, scans, mean_rgb, ins_sem_mapping):
    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'val']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb)

    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = cfg.max_distractors if split == 'train' else cfg.max_test_objects - 1

        if split == 'val':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            logging.info("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            logging.info("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            logging.info("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

        dataset = Nr3dDataset(references=d_set,
                              scans=scans,
                              points_per_object=cfg.points_per_object,
                              max_distractors=max_distractors,
                              class_to_idx=class_to_idx,
                              object_transformation=object_transformation,
                              ins_sem_mapping=ins_sem_mapping,
                              train=True if split == 'train' else False,
                              use_sem=cfg.use_semantic,
                              clip_info=cfg.clip_info_path,
                              clip_dim=cfg.clip_dim)

        seed = None
        if split == 'val':
            seed = 2025
        if split == 'train' and len(dataset) % cfg.batch_size == 1:
            print('dropping last batch during training')
            drop_last = True
        else:
            drop_last = False
        shuffle = split == 'train'

        worker_init_fn = lambda x: np.random.seed(seed)

        data_loaders[split] = DataLoader(dataset,
                                         batch_size=cfg.batch_size,
                                         num_workers=cfg.n_workers,
                                         shuffle=shuffle,
                                         drop_last=drop_last,
                                         pin_memory=False,
                                         worker_init_fn=worker_init_fn)
    return data_loaders


def make_test_data_loader(cfg, referit_data, class_to_idx, scans, mean_rgb, ins_sem_mapping):
    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb)
    max_distractors = cfg.max_test_objects - 1

    def multiple_targets_utterance(x):
        _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
        return len(distractors_ids) > 0

    multiple_targets_mask = referit_data.apply(multiple_targets_utterance, axis=1)
    d_set = referit_data[multiple_targets_mask]
    d_set.reset_index(drop=True, inplace=True)
    print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
    print("removed {} utterances from the test set that don't have multiple distractors".format(
        np.sum(~multiple_targets_mask)))
    print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

    dataset = Nr3dDataset(references=d_set,
                          scans=scans,
                          points_per_object=cfg.points_per_object,
                          max_distractors=max_distractors,
                          class_to_idx=class_to_idx,
                          object_transformation=object_transformation,
                          visualization=True,
                          ins_sem_mapping=ins_sem_mapping,
                          train=False,
                          use_sem=cfg.use_semantic,
                          clip_info=cfg.clip_info_path,
                          clip_dim=cfg.clip_dim)
    seed = cfg.random_seed

    worker_init_fn = lambda x: np.random.seed(seed)

    data_loader = DataLoader(dataset,
                             batch_size=cfg.batch_size,
                             num_workers=cfg.n_workers,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=False,
                             worker_init_fn=worker_init_fn)
    return data_loader
