import torch.nn as nn
import torch


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x.clone()
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        x = self.dropout2(x)
        return self.ln(x + residual)


class CrossFFN(nn.Module):
    def __init__(self, n_head=8, d_model=768, d_hidden=2048, dropout=0.1, use_ffn=True):
        super(CrossFFN, self).__init__()
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=n_head)
        self.dropout_cross = nn.Dropout(dropout)
        self.ln_cross = nn.LayerNorm(d_model)
        # Feed-forward
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = FeedForwardLayer(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

    def forward(self, objs, text):
        objs = objs.transpose(0, 1)
        text = text.transpose(0, 1)
        # Compute cross-attention
        residual = objs.clone()
        objs, _ = self.cross_attn(objs, text, text)
        objs = self.dropout_cross(objs)
        objs = self.ln_cross(objs + residual)
        # Compute feed-forward
        if self.use_ffn:
            objs = self.ffn(objs)

        return objs.transpose(0, 1)

class MultiHeadSpatialNet(nn.Module):
    def __init__(self, n_head, d_model, d_hidden=2048, dropout=0.1):
        super(MultiHeadSpatialNet, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head

        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        # Create a linear layer for each head
        self.score_layers = nn.ModuleList([nn.Linear(self.d_head, 1) for _ in range(n_head)])
        self.ffn = FeedForwardLayer(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

    def forward(self, spatial_features):
        BVN, N, _ = spatial_features.shape
        k = self.k_proj(spatial_features).view(BVN, N, self.n_head, self.d_head)
        v = self.v_proj(spatial_features).view(BVN, N, self.n_head, self.d_head)
        # Initialize containers for the multihead outputs
        multihead_features = []
        multihead_score = []
        for i in range(self.n_head):
            k_head = k[:, :, i, :]
            v_head = v[:, :, i, :]
            score = self.score_layers[i](k_head).squeeze(-1)
            score = nn.functional.softmax(score, dim=-1).unsqueeze(-1)
            feature = (score * v_head).sum(dim=1)
            # feature = score * v_head
            multihead_features.append(feature)
            multihead_score.append(score)
        # Concatenate results from all heads
        feature = torch.cat(multihead_features, dim=-1)
        score = torch.cat(multihead_score, dim=-1).mean(dim=-1)
        features = self.ffn(feature)

        return features


def rotation_aggregate(output):
    B, R, N, _ = output.shape
    scaled_output = output
    return (scaled_output / R).sum(dim=1)


def batch_expansion(tensor, n):
    return tensor.unsqueeze(1).repeat(1, n, *([1] * (tensor.dim() - 1))).view(tensor.size(0) * n, *tensor.shape[1:])


def scale_to_unit_range(x):
    max_x = torch.max(x, dim=-1, keepdim=True).values
    min_x = torch.min(x, dim=-1, keepdim=True).values
    return x / (max_x - min_x + 1e-9)
