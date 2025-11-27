import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from models.backbones.point_net_pp import PointNetTwoBranch
from transformers import BertModel, BertConfig
from models.gat.memgat import GAT_Encoder
from models.gat.matrix_emb import get_pairwise_distance
from torch.autograd import Variable


def get_hybrid_features(net, in_features, clip_features, aggregator=None):
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    out_features2 = []
    for i in range(n_items):
        features, features2 = net(in_features[:, i], clip_features[:, i])
        out_features.append(features)
        out_features2.append(features2)
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
        out_features2 = aggregator(out_features2, dim=independent_dim)
    return out_features, out_features2

class ReferIt3DNet_transformer(nn.Module):

    def __init__(self,
                 cfg,
                 n_obj_classes,
                 ignore_index,
                 class_name_tokens=None,
                 clip_text_feats=None):

        super().__init__()
        self.view_number = cfg.view_number
        self.rotate_number = cfg.rotate_number
        self.points_per_object = cfg.points_per_object
        self.class2idx = cfg.class_to_idx
        self.ignore_index = ignore_index

        self.n_obj_classes = n_obj_classes
        self.class_name_tokens = class_name_tokens
        self.clip_text_feats = clip_text_feats
        self.clip_dim = cfg.clip_dim
        self.clip_align = cfg.clip_align

        self.object_dim = cfg.object_latent_dim
        self.inner_dim = cfg.inner_dim
        self.head_num = cfg.head_num
        self.hidden_dim = cfg.hidden_dim

        self.dropout_rate = cfg.dropout_rate
        self.lang_cls_alpha = cfg.lang_cls_alpha
        self.obj_cls_alpha = cfg.obj_cls_alpha
        self.pp_cls_alpha = cfg.pp_cls_alpha
        self.align_match_alpha = cfg.align_match_alpha

        self.clip_pp_dim = cfg.clip_pp_dim

        self.geo_rel = cfg.geo_rel
        self.geo_dim = cfg.geo_dim

        self.object_encoder = PointNetTwoBranch(sa_n_points=[32, 16, None],
                                          sa_n_samples=[[32], [32], [None]],
                                          sa_radii=[[0.2], [0.4], [None]],
                                          sa_mlps=[[[3, 64, 64, 128]],
                                                   [[128, 128, 128, 256]],
                                                   [[256, 256, self.object_dim, self.object_dim]]])

        self.relation_encoder = GAT_Encoder(lay_num=cfg.lay_number,
                                            gatt_num=cfg.gatt_number,
                                            feat_dim=self.inner_dim,
                                            geo_dim=self.geo_dim,
                                            num_heads=self.head_num,
                                            hidden_dim=self.hidden_dim)

        self.language_encoder = BertModel.from_pretrained(cfg.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:3]

        # Classifier heads
        self.language_target_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim // 2),
                                                 nn.GELU(), nn.Dropout(self.dropout_rate),
                                                 nn.Linear(self.inner_dim // 2, n_obj_classes))

        self.pp_object_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim // 2),
                                           nn.ReLU(), nn.Dropout(self.dropout_rate),
                                           nn.Linear(self.inner_dim // 2, n_obj_classes))
        if not self.clip_align:
            self.object_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim // 2),
                                            nn.ReLU(), nn.Dropout(self.dropout_rate),
                                            nn.Linear(self.inner_dim // 2, n_obj_classes))

        self.object_language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim // 2),
                                                 nn.GELU(), nn.Dropout(self.dropout_rate),
                                                 nn.Linear(self.inner_dim // 2, 1))

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(6, self.inner_dim),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.inner_dim),
        )
        if self.geo_rel:
            self.geo_feature_mapping = nn.Sequential(
                nn.Linear(self.geo_dim, self.inner_dim),
                nn.Dropout(self.dropout_rate),
                nn.LayerNorm(self.inner_dim),
            )

        if self.clip_point:
            self.clip_feature_mapping = nn.Sequential(
                nn.Linear(cfg.clip_dim, self.clip_pp_dim),
                nn.Dropout(self.dropout_rate),
                nn.LayerNorm(self.clip_pp_dim),
            )
        if self.clip_align:
            self.obj2clipdim = nn.Sequential(
                nn.Linear(self.inner_dim, self.clip_dim),
                nn.Dropout(self.dropout_rate),
                nn.LayerNorm(self.clip_dim),
            )

            self.text2clipdim = nn.Sequential(
                nn.Linear(self.inner_dim, self.clip_dim),
                nn.Dropout(self.dropout_rate),
                nn.LayerNorm(self.clip_dim),
            )
            self.match_loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.logit_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def get_bsize(self, r, bsize):
        B = r.shape[0]
        new_size = bsize.clone()
        for bi in range(B):
            if r[bi] == 1 or r[bi] == 3:
                new_size[bi, :, 0] = bsize[bi, :, 1]
                new_size[bi, :, 1] = bsize[bi, :, 0]
        return new_size

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
        input_points = input_points.float().to(self.device)  # (B,N,1024,7)
        box_infos = box_infos.float().to(self.device)  # (B,N,6) cx,cy,cz,lx,ly,lz
        xyz = input_points[..., :3]  # (B,N,1024,3)
        rgb = input_points[..., 3:]
        bxyz = box_infos[:, :, :3]  # B,N,3
        bsize = box_infos[:, :, -3:]
        B, N, P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor([i * 2.0 * np.pi / self.rotate_number for i in range(self.rotate_number)]).to(
            self.device)
        view_theta_arr = torch.Tensor([i * 2.0 * np.pi / self.view_number for i in range(self.view_number)]).to(
            self.device)

        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            r = torch.randint(0, self.rotate_number, (B,))
            theta = rotate_theta_arr[r]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).to(self.device)[
                None].repeat(B, 1, 1)
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[..., :3] = torch.matmul(xyz.reshape(B, N * P, 3), rotate_matrix).reshape(B, N, P, 3)
            bxyz = torch.matmul(bxyz.reshape(B, N, 3), rotate_matrix).reshape(B, N, 3)
            bsize = self.get_bsize(r, bsize)

        # multi-view
        boxs = []
        for r, theta in enumerate(view_theta_arr):
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                          [math.sin(theta), math.cos(theta), 0.0],
                                          [0.0, 0.0, 1.0]]).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B * N, 3), rotate_matrix).reshape(B, N, 3)
            new_size = self.get_bsize(torch.zeros((B,)) + r, bsize)
            boxs.append(torch.cat([rxyz, new_size], dim=-1))

        boxs = torch.stack(boxs, dim=1)  # (B,view_num,N,4)
        if self.view_number == 1:
            boxs = torch.squeeze(boxs, 1)
        return input_points, boxs

    def compute_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, PP_LOGITS):
        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        if PP_LOGITS is not None:
            pp_clf_loss = self.class_logits_loss(PP_LOGITS.transpose(2, 1), batch['class_labels'])
        else:
            pp_clf_loss = 0.0
        lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])

        total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss + self.lang_cls_alpha * lang_clf_loss + self.pp_cls_alpha * pp_clf_loss
        return total_loss

    def align_clip(self, clip_obj_feats, clip_text_feats, obj_feats, text_feats, class_labels, realnums):
        B, N, C = obj_feats.shape
        c_obj_feats = self.obj2clipdim(obj_feats)  # B,N,C
        c_text_feats = self.text2clipdim(text_feats)  # n,C
        norm_obj_feats = F.normalize(c_obj_feats, dim=-1)  # B,N,C
        norm_text_feats = F.normalize(c_text_feats, dim=-1)[None].repeat(B, 1, 1)  # B,n,C

        norm_clip_image_feats = F.normalize(clip_obj_feats, dim=-1)  # B,N,C
        norm_clip_text_feats = F.normalize(clip_text_feats, dim=-1)[None].repeat(B, 1, 1)  # B,n,C

        # correlation loss
        sim_logits_obj2image = torch.bmm(norm_obj_feats, norm_clip_image_feats.transpose(1, 2))  # B,N,N
        sim_labels_obj2image = torch.arange(N).unsqueeze(0).repeat(B, 1).to(sim_logits_obj2image.device)
        for bi in range(B):
            sim_labels_obj2image[bi, realnums[bi]:] = -1
        sim_loss_obj2image = self.match_loss(sim_logits_obj2image.transpose(2, 1), sim_labels_obj2image)

        sim_logits_t2t = torch.mm(norm_text_feats[0], norm_clip_text_feats[0].transpose(0, 1))  # n,n
        sim_labels_t2t = torch.arange(norm_text_feats.shape[1]).to(sim_logits_t2t.device)
        sim_loss_t2t = self.match_loss(sim_logits_t2t, sim_labels_t2t)

        # cross_class
        obj2cliptext = torch.bmm(norm_obj_feats, norm_clip_text_feats.transpose(1, 2))  # B,N,N
        obj2cliptext_loss = self.class_logits_loss(obj2cliptext.transpose(2, 1), class_labels)
        image2text = torch.bmm(norm_clip_image_feats, norm_text_feats.transpose(1, 2))  # B,N,N
        image2text_loss = self.class_logits_loss(image2text.transpose(2, 1), class_labels)

        # class
        obj2text = torch.bmm(norm_obj_feats, norm_text_feats.transpose(1, 2))  # B,N,N

        loss = sim_loss_obj2image + sim_loss_t2t + obj2cliptext_loss + image2text_loss

        return loss, obj2text

    def get_adj_matrix(self, B, N, real_nums, mask):
        adj_mats = torch.zeros(B, N, N, 1)
        for bi in range(B):
            if mask is not None:
                indx = torch.nonzero(mask[bi]).ravel()
                t1 = indx[:, None].repeat(1, indx.shape[0]).ravel()
                t2 = indx[None].repeat(indx.shape[0], 1).ravel()
                adj_mats[bi, t1, t2] = 1
                adj_mats[bi, t2, t1] = 1
            else:
                adj_mats[bi, :real_nums[bi], :real_nums[bi]] = torch.ones((real_nums[bi], real_nums[bi], 1))
            # adj_mats[bi, torch.arange(N), torch.arange(N)] = 0
        return Variable(adj_mats).float()

    def forward(self, batch: dict):
        self.device = self.obj_feature_mapping[0].weight.device

        # rotation augmentation and multi_view generation
        obj_points, boxes = self.aug_input(batch['objects'], batch['box_info'])
        B, R, N = boxes.shape[:3]

        label_text_feats = self.language_encoder(**self.class_name_tokens)[0][:, 0]

        # obj_encoding
        clip_feats = self.clip_feature_mapping(batch['clip_feats'].float())
        obj_feats, aux_feats = get_hybrid_features(self.object_encoder, obj_points, clip_feats,
                                                   aggregator=torch.stack)

        obj_feats = self.obj_feature_mapping(obj_feats)

        if self.clip_align:
            align_loss, CLASS_LOGITS2 = self.align_clip(batch['clip_feats'].float(), self.clip_text_feats,
                                                        obj_feats,
                                                        label_text_feats,
                                                        batch['class_labels'], batch['context_size'])
        else:
            align_loss = 0.0
            CLASS_LOGITS2 = self.object_clf(obj_feats)

        obj_feats = obj_feats + aux_feats
        PP_LOGITS = self.pp_object_clf(obj_feats)

        ## language_encoding
        lang_tokens = batch['lang_tokens']
        lang_infos = self.language_encoder(**lang_tokens)[0]  # (B,len,_)
        # <LOSS>: lang_cls
        LANG_LOGITS = self.language_target_clf(lang_infos[:, 0])
        _, lang_label = torch.max(F.softmax(LANG_LOGITS, -1), dim=-1)

        _, cl_label = torch.max(F.softmax(CLASS_LOGITS2, -1), dim=-1)
        mask = cl_label == lang_label[:, None].repeat(1, N)

        real_nums = batch['context_size']
        for bi in range(B):
            rc = batch['related_class'][bi]
            if rc != '':
                rc_ids = [self.class2idx[cname] for cname in rc.split(',') if cname in self.class2idx]
                for id in rc_ids:
                    mask[bi] = mask[bi] + (cl_label[bi] == id)
            mask[bi, real_nums[bi]:] = False

        # box encoding
        box_embedding = self.box_feature_mapping(boxes)  # (B,R,N,_)

        # gat_encoding
        adj_matrix = self.get_adj_matrix(B, N, real_nums, mask).to(self.device)
        if self.geo_rel:
            geo_feats = get_pairwise_distance(boxes.reshape(B * R, N, -1))
            geo_feats = self.geo_feature_mapping(geo_feats)
        else:
            geo_feats = None

        final_feats, final_att = self.relation_encoder(obj_feats, adj_matrix, geo_feats, lang_infos, box_embedding)

        LOGITS = self.object_language_clf(final_feats).squeeze(-1)

        # <LOSS>
        LOSS = self.compute_loss(batch, CLASS_LOGITS2, LANG_LOGITS, LOGITS, PP_LOGITS)
        LOSS = LOSS + self.align_match_alpha * align_loss

        return LOSS, CLASS_LOGITS2, LANG_LOGITS, LOGITS, final_att
