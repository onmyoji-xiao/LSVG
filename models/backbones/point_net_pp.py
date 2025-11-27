import torch
from torch import nn, Tensor
from external_tools.pointnet2.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG


def break_up_pc(pc: Tensor) -> [Tensor, Tensor]:
    """
    Split the pointcloud into xyz positions and features tensors.
    This method is taken from VoteNet codebase (https://github.com/facebookresearch/votenet)

    @param pc: pointcloud [N, 3 + C]
    :return: the xyz tensor and the feature tensor
    """
    xyz = pc[..., 0:3].contiguous()
    features = (
        pc[..., 3:].transpose(1, 2).contiguous()
        if pc.size(-1) > 3 else None
    )
    return xyz, features


class PointNetTwoBranch(nn.Module):
    """
    Pointnet++ encoder.
    For the hyper parameters please advise the paper (https://arxiv.org/abs/1706.02413)
    """

    def __init__(self, sa_n_points: list,
                 sa_n_samples: list,
                 sa_radii: list,
                 sa_mlps: list,
                 bn=True,
                 use_xyz=True):
        super().__init__()

        n_sa = len(sa_n_points)
        if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
            raise ValueError('Lens of given hyper-params are not compatible')
        self.encoder = nn.ModuleList()

        for i in range(n_sa):
            self.encoder.append(PointnetSAModuleMSG(
                npoint=sa_n_points[i],
                nsamples=sa_n_samples[i],
                radii=sa_radii[i],
                mlps=sa_mlps[i],
                bn=bn,
                use_xyz=use_xyz,
            ))
        out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
        self.fc = nn.Linear(out_n_points * sa_mlps[-1][-1][-1], sa_mlps[-1][-1][-1])

        self.encoder2 = nn.ModuleList()
        self.t = 3
        for i in range(n_sa - (self.t-1)):
            self.encoder2.append(PointnetSAModuleMSG(
                npoint=sa_n_points[i + self.t-1],
                nsamples=sa_n_samples[i + self.t-1],
                radii=sa_radii[i + self.t-1],
                mlps=sa_mlps[i + self.t-1],
                bn=bn,
                use_xyz=use_xyz,
            ))
        self.fc2 = nn.Linear(out_n_points * sa_mlps[-1][-1][-1], sa_mlps[-1][-1][-1])

    def forward(self, features, clip_feats):
        """
        @param features: B x N_objects x N_Points x 3 + C
        """
        xyz, features = break_up_pc(features)
        # xyz (B,1024,3) features (B,C-3,1024)
        features2 = None
        xyz2 = None
        for i in range(len(self.encoder)):
            if i == self.t - 1:
                features2 = features.clone()
                xyz2 = xyz.clone()
            xyz, features = self.encoder[i](xyz, features)

        if features2 is None:
            features2 = features.clone()

        clip_feats = clip_feats[:, :, None].repeat(1, 1, features2.shape[-1])  # (B,128,32)
        features2 = features2 + clip_feats

        for i in range(len(self.encoder2)):
            xyz2, features2 = self.encoder2[i](xyz2, features2)

        return self.fc(features.view(features.size(0), -1)), self.fc2(features2.view(features2.size(0), -1))
