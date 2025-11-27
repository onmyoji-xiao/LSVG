import argparse
import os.path
import os
import time
import shutil
import pprint

gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
from torch import optim
from data.load_data import *
from tools.utils import *
from transformers import BertTokenizer
from tensorboardX import SummaryWriter
from tools.train_api import single_epoch_train, evaluate_on_dataset
from omegaconf import OmegaConf
from dataset.nr3d_dataset import make_data_loaders
from models.refer_net import ReferIt3DNet_transformer
import open_clip

def parse_arguments():
    parser = argparse.ArgumentParser(description='ReferIt3D training')
    parser.add_argument('--cfg', type=str, default="./configs/nr3d.yaml")
    parser.add_argument('--project', type=str, default="base")
    parser.add_argument('--res-dir', type=str, default='./runs',
                        help='where to save training-progress, model, etc')
    parser.add_argument('--log-iter', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=2025)
    parser.add_argument('--resume-path', type=str, default='', help='model-path to resume')

    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)

    args.tb_path = os.path.join(args.res_dir, 'logs', args.project)
    args.save_dir = os.path.join(args.res_dir, args.project)
    if args.resume_path is None:
        if os.path.exists(args.tb_path):
            shutil.rmtree(args.tb_path)
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = create_logger(args.save_dir)
    args_string = pprint.pformat(cfg)
    logger.info(args_string)
    if os.path.exists(args.save_dir + '/models'):
        shutil.rmtree(args.save_dir + '/models')
        shutil.rmtree(args.save_dir + '/dataset')
    shutil.copytree('./models', args.save_dir + '/models')
    shutil.copytree('./dataset', args.save_dir + '/dataset')
    shutil.copy(args.cfg, args.save_dir + '/config.yaml')
    shutil.copy('./train.py', args.save_dir + '/train.py')
    return args, cfg, logger


if __name__ == '__main__':
    # Parse arguments
    args, cfg, logger = parse_arguments()

    device = torch.device('cuda')
    seed_training_code(args.random_seed)

    with open(f'./data/mappings/{cfg.data_name}_classes_mapping.json') as fin:
        ins_sem_mapping = json.load(fin)

    all_scans_in_dict, class_to_idx, referit_data = load_filtered_data(cfg.scannet_file,
                                                                       [cfg.refer_train_file, cfg.refer_val_file],
                                                                       cfg.use_semantic,
                                                                       ins_sem_mapping=ins_sem_mapping)
    # drop unused scans
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    training_scan_ids = set(referit_data[referit_data['is_train']]['scan_id'])

    print('{} training scans will be used.'.format(len(training_scan_ids)))
    mean_rgb = mean_color(training_scan_ids, all_scans_in_dict)
    logger.info(mean_rgb)

    data_loaders = make_data_loaders(cfg, referit_data, class_to_idx, all_scans_in_dict, mean_rgb, ins_sem_mapping)

    label_texts = []
    for key in list(class_to_idx.keys())[:-1]:
        if key != 'pad':
            label_texts.append('The object is ' + key)

    tokenizer = BertTokenizer.from_pretrained(cfg.bert_pretrain_path)
    class_name_tokens = tokenizer(label_texts, return_tensors='pt', padding=True)
    for name in class_name_tokens.data:
        class_name_tokens.data[name] = class_name_tokens.data[name].to(device)

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained=cfg.clip_path)
    clip_model = clip_model.eval().to(device)
    with torch.no_grad():
        tokenized = open_clip.tokenize(label_texts)
        clip_text_feats = clip_model.encode_text(tokenized.to(device))

    pad_idx = class_to_idx['pad']

    cfg.class_to_idx = class_to_idx
    model = ReferIt3DNet_transformer(cfg, len(class_to_idx.keys()) - 1, ignore_index=pad_idx,
                                     class_name_tokens=class_name_tokens, clip_text_feats=clip_text_feats)

    gpu_num = len(gpu.strip(',').split(','))
    if gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=range(gpu_num))
    else:
        model = model.to(device)

    init_lr = cfg.lr
    param_list = [
        {'params': model.object_encoder.parameters(), 'lr': init_lr},
        {'params': model.relation_encoder.parameters(), 'lr': init_lr * 0.1},

        {'params': model.language_encoder.parameters(), 'lr': init_lr * 0.1},
        {'params': model.language_target_clf.parameters(), 'lr': init_lr},
        {'params': model.pp_object_clf.parameters(), 'lr': init_lr},
        {'params': model.object_language_clf.parameters(), 'lr': init_lr},

        {'params': model.obj_feature_mapping.parameters(), 'lr': init_lr},
        {'params': model.box_feature_mapping.parameters(), 'lr': init_lr},

    ]

    if cfg.clip_pp:
        param_list.append({'params': model.clip_feature_mapping.parameters(), 'lr': init_lr})

    if cfg.clip_align:
        param_list.append({'params': model.obj2clipdim.parameters(), 'lr': init_lr})
        param_list.append({'params': model.text2clipdim.parameters(), 'lr': init_lr})
    else:
        param_list.append({'params': model.object_clf.parameters(), 'lr': init_lr})

    if cfg.geo_rel:
        param_list.append({'params': model.geo_feature_mapping.parameters(), 'lr': init_lr})

    optimizer = optim.Adam(param_list, lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 40, 50, 60, 70, 80, 90], gamma=0.65)

    start_epoch = 1
    best_test_acc = 0.0

    if args.resume_path:
        checkpoint = torch.load(args.resume_path)
        loaded_epoch = checkpoint['epoch']
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = loaded_epoch + 1
        print('Resuming from epoch %d' % (start_epoch))
        del checkpoint

    writer = SummaryWriter(args.tb_path)
    logger.info('Starting the training.')

    for epoch in range(start_epoch, cfg.max_epoch + 1):
        t1 = time.time()
        train_meters = single_epoch_train(model, data_loaders['train'], optimizer, device, cfg=cfg, tokenizer=tokenizer, epoch=epoch, logger=logger)
        t2 = time.time()
        test_meters = evaluate_on_dataset(model, data_loaders['val'], device, pad_idx, cfg=cfg,
                                          tokenizer=tokenizer)
        t3 = time.time()

        eval_acc = test_meters['test_referential_acc']
        lr_scheduler.step()

        logger.info('--- epoch %d end ---' % epoch)
        logger.info(
            'train time %f eval_time %f  referential_acc: %f  object_cls_acc: %f  txt_cls_acc: %f' % (
                t2 - t1, t3 - t2, test_meters['test_referential_acc'], test_meters['test_object_cls_acc'],
                test_meters['test_txt_cls_acc']))
        logger.info('--------------------')
        lr = [x['lr'] for x in optimizer.param_groups]
        writer.add_scalar('train/loss', train_meters['train_total_loss'], epoch)
        writer.add_scalar('train/lr', lr[0], epoch)

        writer.add_scalar('val/referential_acc', test_meters['test_referential_acc'], epoch)
        writer.add_scalar('val/object_cls_acc', test_meters['test_object_cls_acc'], epoch)
        writer.add_scalar('val/txt_cls_acc', test_meters['test_txt_cls_acc'], epoch)

        # save
        torch.save({
            'epoch': epoch,
            'model': model.state_dict() if gpu_num == 1 else model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'acc': eval_acc
        }, args.save_dir + '/ckpt_last.pth')

        if eval_acc >= best_test_acc:
            best_test_acc = eval_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict() if gpu_num == 1 else model.module.state_dict(),
                'acc': best_test_acc
            }, args.save_dir + '/ckpt_best.pth')

