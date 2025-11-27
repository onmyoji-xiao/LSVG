import torch.nn as nn
import torch
import math
from models.layers import MultiHeadSpatialNet, CrossFFN


class GraphSelfAttentionLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim=2048, num_heads=8):
        """ Attetion module with vectorized version

        Args:
            position_embedding: [num_rois, nongt_dim, pos_emb_dim]
                                used in implicit relation
            pos_emb_dim: set as -1 if explicit relation
            nongt_dim: number of objects consider relations per image
            fc_dim: should be same as num_heads
            feat_dim: dimension of roi_feat
            num_heads: number of attention heads
            m: dimension of memory matrix
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GraphSelfAttentionLayer, self).__init__()
        # multi head
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(self.feat_dim, self.feat_dim)
        self.key = nn.Linear(self.feat_dim, self.feat_dim)
        self.value = nn.Linear(self.feat_dim, self.feat_dim)
        self.linear_out_ = nn.Conv2d(in_channels=self.num_heads * self.feat_dim,
                                     out_channels=self.feat_dim,
                                     kernel_size=(1, 1),
                                     groups=self.num_heads)

    def forward(self, obj_feats, cross_feats, adj_matrix, label_biases_att):
        """
        Args:
            obj_feats: [B, N, feat_dim]
            adj_matrix: [B, N, N]
            position_embedding: [N, N, pos_emb_dim]
            text_feats: [B, len, feat_dim]
        Returns:
            output: [B, N, output_dim]
        """
        B, N = obj_feats.shape[:2]

        # Q
        q_data = self.query(obj_feats)
        q_data_batch = q_data.view(B, N, self.num_heads, self.head_dim)
        q_data_batch = torch.transpose(q_data_batch, 1, 2)  # [B,num_heads, N, head_dim]
        #
        # K
        k_data = self.key(cross_feats)
        k_data_batch = k_data.view(B, N, self.num_heads, self.head_dim)
        k_data_batch = torch.transpose(k_data_batch, 1, 2)  # [B, num_heads,N, head_dim]

        # V
        v_data = self.value(cross_feats)

        att = torch.matmul(q_data_batch, torch.transpose(k_data_batch, 2, 3))  # [B, num_heads, N, N]
        att = (1.0 / math.sqrt(float(self.head_dim))) * att
        weighted_att = att.transpose(1, 2)  # (B,N,num_heads,N)

        if adj_matrix is not None:
            weighted_att = weighted_att.transpose(2, 3)  # [B,N, N, num_heads]
            zero_vec = -9e15 * torch.ones_like(weighted_att)

            adj_matrix = adj_matrix.view(adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2], 1)
            adj_matrix_expand = adj_matrix.expand((-1, -1, -1, weighted_att.shape[-1]))
            weighted_att_masked = torch.where(adj_matrix_expand > 0, weighted_att, zero_vec)

            weighted_att_masked = weighted_att_masked + label_biases_att.unsqueeze(-1)

            weighted_att = weighted_att_masked
            weighted_att = weighted_att.transpose(2, 3)

        # aff_softmax, [B, N, num_heads, N]
        att_softmax = nn.functional.softmax(weighted_att, 3)
        aff_softmax_reshape = att_softmax.view((B, N, self.num_heads, -1))

        output_t = torch.matmul(aff_softmax_reshape.reshape(B, N * self.num_heads, -1),
                                v_data)  # (B,N*num_heads,N)*(B,N,768)
        output_t = output_t.view((-1, self.num_heads * self.feat_dim, 1, 1))
        linear_out = self.linear_out_(output_t)
        output = linear_out.view((B, N, self.feat_dim))

        return output, att_softmax.sum(dim=2) / self.num_heads


class GAT_Encoder(nn.Module):
    def __init__(self, lay_num, gatt_num, feat_dim, hidden_dim=2048, dropout=0.15,
                 num_heads=16, geo_dim=-1):
        super(GAT_Encoder, self).__init__()
        self.lay_num = lay_num
        self.gatt_num = gatt_num
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(dropout)
        self.feat_fc = nn.Linear(feat_dim, feat_dim, bias=True)
        self.bias = nn.Linear(1, 1, bias=True)
        self.geo_dim = geo_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.g_atts = nn.ModuleList([GraphSelfAttentionLayer(hidden_dim=self.hidden_dim,
                                                             num_heads=num_heads,
                                                             feat_dim=self.feat_dim)
                                     for _ in range(gatt_num)])

        self.sp_agg = nn.ModuleList([MultiHeadSpatialNet(n_head=self.num_heads, d_model=self.feat_dim,
                                                         d_hidden=self.hidden_dim, dropout=dropout)
                                     for _ in range(lay_num)])

        self.obj_crossattns = nn.ModuleList([CrossFFN(n_head=num_heads, d_model=self.feat_dim, d_hidden=self.hidden_dim,
                                                      dropout=dropout)
                                             for _ in range(lay_num)])

        self.lay_num = lay_num

    def forward(self, obj_feats, adj_matrix, geo_feats, text_feats, box_embedding):
        B, R, N = box_embedding.shape[:3]
        obj_feats = self.feat_fc(self.dropout(obj_feats))  # (B,N,768)
        v_text_feats = text_feats[:, None].repeat(1, R, 1, 1).reshape(B * R, -1, self.feat_dim)  # (B*R,len,768)
        att = None
        input_adj_matrix = adj_matrix[:, None].repeat(1, R, 1, 1, 1).reshape(B * R, N, N, 1)
        v_biases_neighbors = self.bias(input_adj_matrix).squeeze(-1)  # (B,N,N)

        obj_feats = (obj_feats[:, None] + box_embedding).reshape(B * R, -1, self.feat_dim)  # (B*R,N,768)
        for i in range(self.lay_num):
            self_feats = self.obj_crossattns[i](obj_feats, v_text_feats)  # obj-text

            fuse_feats = self_feats[:, None].repeat(1, N, 1, 1) + geo_feats  #

            dec_feats = self.sp_agg[i](fuse_feats.reshape(B * R * N, N, -1)).reshape(B * R, N, -1)
            if i < self.gatt_num:
                re_feats, att = self.g_atts[i](dec_feats, dec_feats, input_adj_matrix, v_biases_neighbors)
                att = att.reshape(B, R, N, N).sum(dim=1) / R
                obj_feats = (self_feats + re_feats.reshape(B * R, N, -1))
            else:
                obj_feats = (self_feats + dec_feats.reshape(B * R, N, -1))

        return obj_feats.reshape(B, R, N, -1).sum(dim=1) / R, att
