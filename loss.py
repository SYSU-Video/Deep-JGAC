import torch
import MinkowskiEngine as ME

from data_utils import isin, istopk
criterion = torch.nn.BCEWithLogitsLoss()
from nndistance.modules.nnd import NNDModule
nndistance = NNDModule()
crt = torch.nn.MSELoss()


def rgb_loss( net_out,net_in):
    """Compute the xyz-loss."""

    dist=crt(net_out.F,net_in.F)
    # loss = torch.mean(dist)
    return dist
def rgb_loss_test( net_out,net_in):
    """Compute the xyz-loss."""
    x_hat_G,x_hat_F=net_out.decomposed_coordinates_and_features
    gt_G,gt_F=net_in.decomposed_coordinates_and_features

    batch_size = len(x_hat_F)
    # dist = torch.zeros(1, device=x_hat_G[0].device)
    gt_color_all=[]
    for i in range(batch_size):
        _, _, ind1, ind2 = nndistance(
            x_hat_G[i].unsqueeze(0).float().contiguous(),
            gt_G[i].unsqueeze(0).float().contiguous()
        )
        gt_color=(gt_F[i])[ind1.squeeze(0)]

        linshi_geo=(gt_G[i])[ind1.squeeze(0)]
        mask=torch.norm(linshi_geo.float()-x_hat_G[i].float(),dim=1)
        indices=torch.nonzero(mask==0)
        gt_color_all.append(gt_color)
    gt_color_all=torch.cat(gt_color_all,0)
    x_hat_F=torch.cat(x_hat_F,0)
    dist=crt(gt_color_all,x_hat_F)
    # loss = torch.mean(dist)
    return dist
def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C)
    bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0]
    
    return sum_bce

def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits

def get_metrics(data, groud_truth):
    mask_real = isin(data.C, groud_truth.C)
    nums = [len(C) for C in groud_truth.decomposed_coordinates]
    mask_pred = istopk(data, nums, rho=1.0)
    metrics = get_cls_metrics(mask_pred, mask_real)

    return metrics[0]

def get_cls_metrics(pred, real):
    TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]
    TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]

