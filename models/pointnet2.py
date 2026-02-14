import torch

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    # group_idx[sqrdists > radius ** 2] = N
    v,idx=torch.sort(sqrdists)
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # mask = group_idx == N
    # group_idx[mask] = group_first[mask]
    idx=idx[:,:,:nsample]
    return idx
def DOUBLE_query_ball_point(nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, M,S, _ = new_xyz.shape
    group_idx_all = torch.zeros([B,M,S,nsample],dtype=torch.int64)
    for i in range(M):
        # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        # sqrdists = square_distance(new_xyz[:,i,:,:], xyz)
        # group_idx[sqrdists > radius ** 2] = N
        # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        # mask = group_idx == N
        # group_idx[mask] = group_first[mask]
        sqrdists = square_distance(new_xyz[:,i,:,:], xyz)
        # group_idx[sqrdists > radius ** 2] = N
        v, idx = torch.sort(sqrdists)
        # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        # mask = group_idx == N
        # group_idx[mask] = group_first[mask]
        idx = idx[:, :, :nsample]
        group_idx_all[:,i,:,:]=idx
    return group_idx_all

def sample_and_group(npoint,  nsample, xyz, points,nsample1=3,returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point( nsample, xyz, new_xyz)
    # idx1 = query_ball_point(radius, 3, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    idx1=DOUBLE_query_ball_point(nsample1,xyz,grouped_xyz)
    grouped_xyz1 = index_points(xyz, idx1)
    # grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_xyz_norm1 = grouped_xyz1 - grouped_xyz.view(B, S, nsample,1, C)

    if points is not None:
        grouped_points = index_points(points, idx1)
        new_points = torch.cat([grouped_xyz1,grouped_xyz_norm1, grouped_points], dim=-1) # [B, npoint, nsample,nsample1, C+D]
    else:
        new_points = grouped_xyz1
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points,fps_idx


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points