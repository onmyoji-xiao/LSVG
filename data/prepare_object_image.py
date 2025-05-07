import torch
from PIL import Image
import open_clip
import json
import os
import argparse
import h5py
import numpy as np


def get_scene_object_feats(top_dir,mapping):
    object_ids = []
    object_crop_images = []
    for obj_id in mapping:
        box_lists = mapping[obj_id]
        boxes = np.array(box_lists)[:, 1:].astype(np.int64)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        inds = np.argsort(areas)

        frame_name1, xmin1, ymin1, xmax1, ymax1 = box_lists[inds[-1]]
        img1 = Image.open(os.path.join(top_dir, 'color', frame_name1 + '.jpg'))
        img1 = img1.crop((xmin1, ymin1, xmax1, ymax1))
        resized_image = np.array(img1.resize((224, 224)))/255.0

        object_crop_images.append(resized_image)
        object_ids.append(int(obj_id))

    return np.stack(object_crop_images), object_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data process')

    parser.add_argument('-scans_dir', type=str, default='/home/bip/xhb/xf/3dvg/scannet/scannet_compress',
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('--save_dir', default='', type=str, help='preprocess data path')
    args = parser.parse_args()

    device = torch.device('cuda:7')

    with h5py.File(os.path.join(args.save_dir, 'object_image.hdf5'), "a") as f:
        for scan_id in os.listdir(args.scans_dir):
            map_path = os.path.join(args.scans_dir, scan_id, 'object_image_mapping.json')
            if not os.path.exists(map_path):
                print(scan_id, ' not found mapping')
                continue
            with open(map_path, 'r') as file:
                data = json.load(file)
            images, object_ids = get_scene_object_feats(os.path.join(args.scans_dir, scan_id),data)

            g = f.create_group(scan_id)
            g.create_dataset('object_ids', data=object_ids, compression="gzip")
            g.create_dataset('object_images', data=images, compression="gzip")
            print(scan_id, ' done')

# from spellchecker import SpellChecker
# spell = SpellChecker()
# misspelled_words = spell.unknown(ori_tokens)
# for word in misspelled_words:
#     if spell.correction(word) in self.mapping:
#         print("Misspelled words: ",word, spell.correction(word))
#         tokens = tokens.replace(word, spell.correction(word))