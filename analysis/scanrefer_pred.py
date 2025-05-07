import pandas as pd

from .utterances import is_explicitly_view_dependent
from dataset.utils import decode_stimulus_string
from torch.utils.data import DataLoader
from tools.test_api import detailed_predictions_on_dataset
import numpy as np
import tqdm
import torch
from dataset.cuboid import iou_3d


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ['objects', 'tokens', 'target_pos']  # all models use these

    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if args.lang_cls_alpha > 0:
        batch_keys.append('target_class')

    batch_keys.append('clip_feats')
    return batch_keys


def deep_predictions(model, dataset, device, args, out_file, tokenizer):
    metrics = dict()
    test_seeds = [args.random_seed, 1, 10, 20, 100]
    overall_25_acc = []
    overall_50_acc = []
    unique_25_acc = []
    unique_50_acc = []
    multiple_25_acc = []
    multiple_50_acc = []
    for seed in test_seeds:
        worker_init_fn = lambda x: np.random.seed(seed)
        d_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=False,
                              worker_init_fn=worker_init_fn)

        batch_keys = make_batch_keys(args)

        ious = []
        masks = []
        for batch in tqdm.tqdm(d_loader):
            # Move data to gpu
            for k in batch_keys:
                if k not in batch or isinstance(batch[k], list):
                    continue
                batch[k] = batch[k].to(device)

            lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
            for name in lang_tokens.data:
                lang_tokens.data[name] = lang_tokens.data[name].to(device)
            batch['lang_tokens'] = lang_tokens

            # Forward pass
            LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, attmaps = model(batch)
            res = {}
            res['logits'] = LOGITS

            # Update the loss and accuracy meters
            target = batch['target_pos']
            batch_size = target.size(0)  # B x N_Objects

            predictions = torch.argmax(res['logits'], dim=1)

            target_box = batch['target_box'].cpu().numpy()  # (B,6)
            pred_box = batch['box_info'].cpu().numpy()  # (B,52,6)

            for j in range(batch_size):
                pd = pred_box[j][predictions[j]]
                iou = iou_3d(pd, target_box[j])
                ious.append(iou)

            mask = batch['unique_multiple'].cpu().numpy()
            masks.append(mask)

        masks = np.concatenate(masks)
        unique_mask = (masks == 0)

        ious = np.array(ious)

        overall_25_acc.append(ious[ious >= 0.25].shape[0] / ious.shape[0])
        overall_50_acc.append(ious[ious >= 0.5].shape[0] / ious.shape[0])

        unique_ious = ious[unique_mask]
        multiple_ious = ious[~unique_mask]
        unique_25_acc.append(unique_ious[unique_ious >= 0.25].shape[0] / unique_ious.shape[0])
        unique_50_acc.append(unique_ious[unique_ious >= 0.5].shape[0] / unique_ious.shape[0])
        multiple_25_acc.append(multiple_ious[multiple_ious >= 0.25].shape[0] / multiple_ious.shape[0])
        multiple_50_acc.append(multiple_ious[multiple_ious >= 0.5].shape[0] / multiple_ious.shape[0])

        print(unique_50_acc[-1], multiple_50_acc[-1], overall_50_acc[-1])

    metrics['ref_iou_rate_0.25'] = np.array(overall_25_acc).mean()
    metrics['ref_iou_rate_0.5'] = np.array(overall_50_acc).mean()
    metrics['unique_iou_rate_0.25'] = np.array(unique_25_acc).mean()
    metrics['unique_iou_rate_0.5'] = np.array(unique_50_acc).mean()
    metrics['multiple_iou_rate_0.25'] = np.array(multiple_25_acc).mean()
    metrics['multiple_iou_rate_0.5'] = np.array(multiple_50_acc).mean()
    return metrics
