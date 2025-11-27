
import torch


def scale_to_unit_range(x):
    max_x = torch.max(x, dim=-1, keepdim=True).values
    min_x = torch.min(x, dim=-1, keepdim=True).values
    return x / (max_x - min_x + 1e-9)


def get_pairwise_distance(boxes):
    x = boxes[..., :3]
    B, N, _ = x.shape
    relative_positions = x[:, None] - x[:, :, None]
    # Obtain the xy distances
    xy_distances = relative_positions[..., :2].norm(dim=-1, keepdim=True) + 1e-9
    r = xy_distances.squeeze(-1)
    phi = torch.atan2(relative_positions[..., 1], relative_positions[..., 0])  # Azimuth angle
    theta = torch.atan2(r, relative_positions[..., 2])  # Elevation angle
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    relative_positions = torch.cat([relative_positions, xy_distances, sin_phi.unsqueeze(-1), cos_phi.unsqueeze(-1),
                                    sin_theta.unsqueeze(-1), cos_theta.unsqueeze(-1)], dim=-1)
    # Normalize x-y plane to unit vectors
    relative_positions[..., :2] = relative_positions[..., :2] / xy_distances
    # Scale z values so that max(z) - min(z) = 1
    relative_positions[..., 2] = scale_to_unit_range(relative_positions[..., 2])
    # Scale d values between 0 and 1 for each set of relative positions independently.
    relative_positions[..., 3] = scale_to_unit_range(relative_positions[..., 3])

    return relative_positions
