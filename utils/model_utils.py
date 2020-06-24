import torch
import numpy as np
import os
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear as Lin, Sequential as Seq, ReLU, BatchNorm1d, LeakyReLU
from torch_geometric.nn import fps, radius, PointConv


def mlp(channels, last=False, leaky=False):
    if leaky:
        rectifier = LeakyReLU
    else:
        rectifier = Relu
    l = [Seq(Lin(channels[i - 1], channels[i], bias=False), BatchNorm1d(channels[i]), rectifier())
            for i in range(1, len(channels)-1)]
    if last:
        l.append(Seq(Lin(channels[-2], channels[-1], bias=True)))
    else:
        l.append(Seq(Lin(channels[-2], channels[-1], bias=False), BatchNorm1d(channels[-1]), rectifier()))
    return Seq(*l)


def simulate_partial_point_clouds(data, npts, task):
    """
    Simulate partial point clouds.
    """
    pos, batch, label = data.pos, data.batch, data.y

    # noise = torch.randn_like(pos)*0.08
    # pos = pos + noise

    bsize = batch.max() + 1
    pos = pos.view(bsize, -1, 3)
    batch = batch.view(bsize, -1)
    if task == 'segmentation':
        label = label.view(bsize, -1)

    out_pos, out_batch, out_label = [], [], []
    for i in range(pos.size(0)):
        while True:
            # define a plane by its normal and it goes through the origin
            vec = torch.randn(3).to(pos.device)
            # mask out half side of points
            mask = pos[i].matmul(vec) > 0
            p = pos[i][mask]
            if p.size(0) >= 300:
                break

        # mask = pos[i, :, 2]>0
        # if torch.sum(mask) == 0:
        #     mask = pos[i, :, 1]>-0.3
        # if torch.sum(mask) == 0:
        #     mask = pos[i, :, 1]>-0.5
        # p = pos[i][mask]

        # ensure output contains fixed number of points
        idx = np.random.choice(p.size(0), npts, True)
        out_pos.append(pos[i][mask][idx])
        out_batch.append(batch[i][mask][idx])
        if task == 'segmentation':
            out_label.append(label[i][mask][idx])

    data.pos = torch.cat(out_pos, dim=0)
    data.batch = torch.cat(out_batch, dim=0)
    if task == 'segmentation':
        data.y = torch.cat(out_label, dim=0)
    return data


class NormalizeSphere(object):
    """
    Normalize point clouds into a unit sphere
    """
    def __init__(self, center):
        self.is_center = center
        if center:
            self.center = T.Center()
        else:
            self.center = None

    def __call__(self, data):
        if self.center is not None:
            data = self.center(data)

        scale = (1 / data.pos.norm(dim=-1).max()) * 0.999999
        data.pos = data.pos * scale
        return data

    def __repr__(self):
        return '{}(center={})'.format(self.__class__.__name__, self.is_center)


def chamfer_loss(x, y):
    """
    Compute chamfer distance for x and y. Note there are multiple version of chamfer
    distance. The implemented chamfer distance is defined in:

        https://arxiv.org/pdf/1612.00603.pdf.

    It finds the nearest neighbor in the other set and computes their squared
    distances which are summed over both target and ground-truth sets.

    Arguments:
        x: [bsize, m, 3]
        y: [bsize, n, 3]

    Returns:
        dis: [bsize]
    """
    x = x.unsqueeze(1)
    y = y.unsqueeze(2)
    diff = (x - y).norm(dim=-1)
    # diff = (x - y).pow(2).sum(dim=-1)
    dis1 = diff.min(dim=1)[0].mean(dim=1)
    dis2 = diff.min(dim=2)[0].mean(dim=1)
    dis = dis1 + dis2
    return dis


def augment_transforms(args):
    """
    define transformation
    """
    pre_transform = None
    if args.norm == 'scale':
        pre_transform = T.NormalizeScale()
    elif args.norm == 'sphere':
        pre_transform = NormalizeSphere(center=True)
    elif args.norm == 'sphere_wo_center':
        pre_transform = NormalizeSphere(center=False)
    else:
        pass

    transform = []
    if args.dataset == 'shapenet':
        transform.append(T.FixedPoints(args.num_pts))
    if args.dataset == 'modelnet':
        transform.append(T.SamplePoints(args.num_pts))

    # if args.is_randRotY:
    #     transform.append(T.RandomRotate(180, axis=1))
    transform = T.Compose(transform)
    return pre_transform, transform


def create_batch_one_hot_category(category):
    """
    Create batch one-hot vector for indicating category. ShapeNet.

    Arguments:
        category: [batch]
    """
    batch_one_hot_category = np.zeros((len(category), 16))
    for b in range(len(category)):
        batch_one_hot_category[b, int(category[b])] = 1
    batch_one_hot_category = torch.from_numpy(batch_one_hot_category).float().cuda()
    return batch_one_hot_category


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
