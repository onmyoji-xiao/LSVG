import torch
from PIL import Image, ImageDraw, ImageOps
import open_clip
import json
import os
import argparse
import h5py
import numpy as np


def clip_match(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    sim = image_features @ text_features.T
    b, c = sim.shape
    sim = (sim / b).sum(dim=0)
    # c_id = torch.argmax(sim, dim=-1)
    values, c_id = torch.topk(sim, k=2)
    return c_id


def fill_pad(image):
    w, h = image.size
    if w > h:
        tmp1 = (w - h) // 2
        tmp2 = w - h - tmp1
        border = (0, tmp1, 0, tmp2)
        image = ImageOps.expand(image, border=border, fill=0)
    elif w < h:
        tmp1 = (h - w) // 2
        tmp2 = h - w - tmp1
        border = (tmp1, 0, tmp2, 0)
        image = ImageOps.expand(image, border=border, fill=0)
    return image


def get_object_cls(top_dir, model, preprocess, mapping, text_features):
    object_ids = []
    object_cls = []
    object_feats = []
    for obj_id in mapping:
        box_lists = mapping[obj_id]
        pnum = np.array([bb[1] for bb in box_lists])
        inds = np.argsort(pnum)[::-1]

        object_crop_images = []
        for i in inds[:1]:
            frame_name = box_lists[i][0]
            xmin, ymin, xmax, ymax = box_lists[i][2], box_lists[i][3], box_lists[i][4], box_lists[i][5]
            # frame_name, _, pps = box_lists[i]
            # pps = np.array(pps)
            # xmin, ymin, xmax, ymax = np.min(pps[:, 0]), np.min(pps[:, 1]), np.max(pps[:, 0]), np.max(pps[:, 1])
            img = Image.open(os.path.join(top_dir, 'color', frame_name + '.jpg'))
            # if ymax - ymin > xmax - xmin:
            #     tmp = (ymax - ymin) - (xmax - xmin)
            #     cxmin = max(0, xmin - tmp // 2)
            #     cxmax = min(img.size[0], xmax + tmp // 2)
            #     cymin = ymin
            #     cymax = ymax
            # else:
            #     tmp = (xmax - xmin) - (ymax - ymin)
            #     cymin = max(0, ymin - tmp // 2)
            #     cymax = min(img.size[1], ymax + tmp // 2)
            #     cxmin = xmin
            #     cxmax = xmax
            # img = img.convert("RGBA")
            # mask = Image.new('L', img.size, 0)
            # draw = ImageDraw.Draw(mask)
            # vertices = [(p[0], p[1]) for p in pps]
            # draw.polygon(vertices, fill=255)
            # img.putalpha(mask)

            cxmin = max(0, xmin - 1)
            cymin = max(0, ymin - 1)
            cxmax = min(img.size[0], xmax + 1)
            cymax = min(img.size[1], ymax + 1)

            # draw = ImageDraw.Draw(img)
            # border_width = 4
            # for i in range(border_width):
            #     draw.rectangle([xmin - i, ymin - i, xmax + i, ymax + i], outline="red")
            img = img.crop((cxmin, cymin, cxmax, cymax))
            img = fill_pad(img)

            try:
                object_crop_images.append(preprocess(img))
            except:
                continue
        if len(object_crop_images) > 0:
            with torch.no_grad():
                image_features = model.encode_image(torch.stack(object_crop_images).to(device))
            c_id = clip_match(image_features, text_features)
            object_ids.append(int(obj_id))
            object_cls.append([int(c_id[0].item()), int(c_id[1].item())])
            object_feats.append(image_features[0])
            # img = img.convert('RGB')
            # img.save(f'./out/{semantics[c_id[0]]}+{semantics[c_id[1]]}_{id2ins[scan_id][obj_id]}_{obj_id}.jpg')

    return object_cls, object_ids, torch.stack(object_feats)


def get_object_feature(top_dir, model, preprocess, mapping):
    object_ids = []
    object_feats = []
    for obj_id in mapping:
        box_lists = mapping[obj_id]
        pnum = np.array([bb[1] for bb in box_lists])
        inds = np.argsort(pnum)[::-1]

        object_crop_images = []
        for i in inds[:10]:
            frame_name, pnum, xmin, ymin, xmax, ymax= box_lists[i]
            # xmin, ymin, xmax, ymax = box
            if pnum < 8:
                break
            img = Image.open(os.path.join(top_dir, 'color', frame_name + '.jpg'))

            cxmin = max(0, xmin - 1)
            cymin = max(0, ymin - 1)
            cxmax = min(img.size[0], xmax + 1)
            cymax = min(img.size[1], ymax + 1)

            img = img.crop((cxmin, cymin, cxmax, cymax))
            img = fill_pad(img)

            try:
                object_crop_images.append(preprocess(img))
            except:
                continue
        if len(object_crop_images) > 0:
            with torch.no_grad():
                image_features = model.encode_image(torch.stack(object_crop_images).to(device))
            object_feats.append(image_features.cpu().numpy())
            object_ids.append(int(obj_id))

    return object_feats, object_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data process')

    parser.add_argument('-scans_dir', type=str, default='/home/xf/codes/DataSet/scannet_rgbd_sample',
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('-clip_path', type=str,
                        default='/home/xf/codes/3dvg/pretrained/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin')
    parser.add_argument('--save_dir', default='', type=str, help='preprocess data path')
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained=args.clip_path)

    device = torch.device('cuda:3')
    model = model.to(device)

    scan_ids = sorted(os.listdir(args.scans_dir))
    scan_ids = [s for s in scan_ids if '_00' in s]

    with h5py.File(os.path.join(args.save_dir, './vith-clip_feats_pad0.hdf5'), "a") as f:
        for scan_id in scan_ids:
            if scan_id in f:
                continue
            # if scan_id not in id2ins:
            #     continue
            map_path = os.path.join(args.scans_dir, scan_id, 'pc_object_box_mapping.json')
            # map_path = os.path.join(args.scans_dir, scan_id, 'object_image_mapping.json')
            if not os.path.exists(map_path):
                print(scan_id, ' not found mapping')
                continue
            with open(map_path, 'r') as file:
                data = json.load(file)

            object_feats, object_ids = get_object_feature(os.path.join(args.scans_dir, scan_id), model, preprocess,
                                                          data)
            g = f.create_group(scan_id)
            for id, ff in zip(object_ids, object_feats):
                g.create_dataset(str(id), data=ff, compression="gzip")
            print(scan_id, ' done')
    #
    # with h5py.File('/home/xf/codes/planB/pre_data/gs2id_scannet_compress.hdf5', "r") as f:
    #     for scan_id in f:
    #
    #         if '_00' not in scan_id or scan_id not in id2ins:
    #             continue
    #         object_image_dict = {}
    #         scan_path = os.path.join(args.scans_dir, scan_id)
    #         depth_files = os.listdir(os.path.join(scan_path, 'depth'))
    #         depth_files.sort(key=lambda e: (int(e.split('.')[0]), e))
    #         id_maps = f[scan_id]['pix2id']
    #         for id_map, filename in zip(id_maps, depth_files):
    #             id2box = {}
    #             pix2id = np.array(id_map)
    #             object_ids = np.unique(pix2id)
    #             for obj_id in object_ids:
    #                 if obj_id == -1:
    #                     continue
    #                 indices = np.where(pix2id == obj_id)
    #                 xmin = indices[1].min()
    #                 ymin = indices[0].min()
    #                 xmax = indices[1].max()
    #                 ymax = indices[0].max()
    #                 info = [filename.split('.')[0], len(indices[0]), int(xmin), int(ymin), int(xmax), int(ymax), id_map]
    #                 if obj_id not in object_image_dict:
    #                     object_image_dict[int(obj_id)] = [info]
    #                 else:
    #                     object_image_dict[int(obj_id)].append(info)
    #         object_ids = []
    #         object_cls = []
    #         for obj_id in object_image_dict:
    #             box_lists = object_image_dict[obj_id]
    #             pnum = np.array([bb[1] for bb in box_lists])
    #             inds = np.argsort(pnum)[::-1]
    #             object_crop_images = []
    #             for i in inds[:2]:
    #                 frame_name, _, xmin, ymin, xmax, ymax, id_map = box_lists[i]
    #                 img = Image.open(os.path.join(scan_path, 'color', frame_name + '.jpg'))
    #                 mask = np.expand_dims((id_map != obj_id), axis=-1)
    #                 mask = np.repeat(mask, 3, axis=-1)
    #                 img = np.array(img)
    #                 img[mask] = 255
    #                 img = Image.fromarray(img).crop((xmin, ymin, xmax, ymax))
    #                 try:
    #                     object_crop_images.append(preprocess(img))
    #                 except:
    #                     continue
    #             if len(object_crop_images) > 0:
    #                 with torch.no_grad():
    #                     image_features = model.encode_image(torch.stack(object_crop_images).to(device))
    #                 c_id = clip_match(image_features, text_features)
    #                 object_ids.append(int(obj_id))
    #                 object_cls.append(int(c_id.item()))
    #                 # if scan_id in id2ins and int(obj_id) in id2ins[scan_id]:
    #                 img.save(f'./out/{semantics[c_id]}_{id2ins[scan_id][str(obj_id)]}_{obj_id}.jpg')
    #
    #         print(scan_id, ' done')

# from spellchecker import SpellChecker
# spell = SpellChecker()
# misspelled_words = spell.unknown(ori_tokens)
# for word in misspelled_words:
#     if spell.correction(word) in self.mapping:
#         print("Misspelled words: ",word, spell.correction(word))
#         tokens = tokens.replace(word, spell.correction(word))
