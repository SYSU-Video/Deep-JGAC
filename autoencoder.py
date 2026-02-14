import time

import torch
import MinkowskiEngine as ME
import gc
from data_utils import isin, istopk
from utils import *
from nndistance.modules.nnd import NNDModule

nndistance = NNDModule()
from data_utils import *
from data_loader import yuv_rgb
from tool import sp2ply
import pytorch3d
from pytorch3d.ops import  knn_points
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
def rgb_get(net_out, net_in,coordinate_manager=None):
    """Compute the xyz-loss."""
    color_in = time.time()
    x_hat_G, x_hat_F = net_out.decomposed_coordinates_and_features
    gt_G, gt_F = net_in.decomposed_coordinates_and_features

    batch_size = len(x_hat_F)
    color_all = []
    for i in range(batch_size):
        # _, _, ind1, ind2 = nndistance(
        #     x_hat_G[i].unsqueeze(0).float().contiguous(),
        #     gt_G[i].unsqueeze(0).float().contiguous()
        # )
        x_nn = knn_points(x_hat_G[i].unsqueeze(0).float().contiguous(), gt_G[i].unsqueeze(0).float().contiguous(), K=1)

        idx = x_nn.idx.squeeze(0)  # [N, K]
        gt_color = gt_F[i][idx]  # [N, K, C]
        gt_color = gt_color.mean(1)  # 对K取平均 → [N, C]
        color_all.append(gt_color)

        # gt_color = (gt_F[i])[x_nn.idx.squeeze(0)]
        # color_all.append(gt_color.squeeze(1))
    color_all = torch.cat(color_all, 0)
    color_out = time.time()
    color_time=color_out-color_in
    tensor_in=time.time()

    # rec_pcd = open3d.geometry.PointCloud()  # 定义点云
    # rec_pcd.points = open3d.utility.Vector3dVector((x_hat_G[0].cpu()).float())  # 定义点云坐标位置[N,3]
    # rec_pcd.colors = open3d.utility.Vector3dVector(yuv_rgb((color_all.cpu())))  # 定义点云坐标位置[N,3]
    #
    # if not os.path.exists('pc_file'): os.makedirs('pc_file')
    # # orifile='pc_file/'+filename+'_ori.ply'
    # recfile = 'pc_file/' + 'basketball_player_vox11_00000200' + '_r' + str(0) + '_rec.ply'
    # open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

    if coordinate_manager==None:
        # color_all = ME.SparseTensor(features=color_all, coordinates=(net_out.C), device=net_out.device)
        coords = x_hat_G[0]  # [N, 3]
        batch_col = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
        coords_with_batch = torch.cat([batch_col, coords], dim=1)  # [N, 4]

        color_all = ME.SparseTensor(features=color_all, coordinates=coords_with_batch, device=net_out.device)
        # color_all = ME.SparseTensor(features=color_all, coordinates=x_hat_G[0], device=net_out.device)
    else:
        color_all = ME.SparseTensor(features=color_all, coordinates=(net_out.C), device=net_out.device,coordinate_manager=coordinate_manager)
    tensor_out=time.time()
    tensor_time=tensor_out-tensor_in
    return color_all, color_time, tensor_time


def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))

    return torch.nn.Sequential(*layers)


class Encoder(torch.nn.Module):
    def __init__(self, channels=[1, 16, 32, 64, 32, 8]):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels[2])

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels[3])

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block2 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels[4])

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.pool = ME.MinkowskiMaxPooling(kernel_size=1, stride=2, dimension=3)




        self.relu = ME.MinkowskiReLU()
        # self.bn1=ME.MinkowskiBatchNorm(128)
        # self.bn2=ME.MinkowskiBatchNorm(128)
        # self.bn3=ME.MinkowskiBatchNorm(128)


        # G decode
        self.decoder_G = Decoder(channels=[8, 64, 32, 16])
        self.score3_predictor_AtoG = PredictorLG(16)
        self.score4_predictor_AtoG = PredictorLG(32)
        self.score5_predictor_AtoG = PredictorLG(64)
        self.score6_predictor_AtoG = PredictorLG(32)
        self.exchange = FeatureExchange()  # (self,x,x1,mask,threshold)
        self.pruning = ME.MinkowskiPruning()
        self.union = ME.MinkowskiUnion()

    def forward(self, x, color):
        torch.cuda.empty_cache()
        gc.collect()

        out0 = ((self.conv0(x)))
        mask = self.score3_predictor_AtoG(out0)
        mask = F.softmax(mask.F, dim=-1)[:, 0]
        out0 = ME.SparseTensor(
            features=out0.F * mask.unsqueeze(1) + out0.F,
            coordinate_map_key=out0.coordinate_map_key,
            coordinate_manager=out0.coordinate_manager,
            device=out0.device)

        out0 = self.relu(self.down0(self.relu((out0))))
        out0 = self.block0(out0)
        mask = self.score4_predictor_AtoG(out0)
        mask = F.softmax(mask.F, dim=-1)[:, 0]
        out0 = ME.SparseTensor(
            features=out0.F * mask.unsqueeze(1) + out0.F,
            coordinate_map_key=out0.coordinate_map_key,
            coordinate_manager=out0.coordinate_manager,
            device=out0.device)

        # second scale
        out1 = self.relu(self.down1(self.relu(self.conv1(out0))))
        out1 = self.block1(out1)

        mask = self.score5_predictor_AtoG(out1)
        mask = F.softmax(mask.F, dim=-1)[:, 0]
        out1 = ME.SparseTensor(
            features=out1.F * mask.unsqueeze(1) + out1.F,
            coordinate_map_key=out1.coordinate_map_key,
            coordinate_manager=out1.coordinate_manager,
            device=out1.device)
        # third sclae
        out2 = self.relu(self.down2(self.relu(self.conv2(out1))))
        out2 = self.block2(out2)

        mask = self.score6_predictor_AtoG(out2)
        mask = F.softmax(mask.F, dim=-1)[:, 0]
        out2 = ME.SparseTensor(
            features=out2.F * mask.unsqueeze(1) + out2.F,
            coordinate_map_key=out2.coordinate_map_key,
            coordinate_manager=out2.coordinate_manager,
            device=out2.device)
        out2 = self.conv3(out2)

        # A
        y_list = [out2, out1, out0]
        y_G = y_list[0]
        y_G_noise = torch.nn.init.uniform_(torch.zeros_like(y_G.F), -0.5, 0.5)
        if self.training:
            compressed_y_G = y_G.F + y_G_noise

        else:
            compressed_y_G = torch.round(y_G.F)

        compressed_y_G = ME.SparseTensor(
            features=compressed_y_G,
            coordinate_map_key=y_G.coordinate_map_key,
            coordinate_manager=y_G.coordinate_manager,
            device=y_G.device)
        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
                     for ground_truth in ground_truth_list]
        torch.cuda.empty_cache()
        gc.collect()

        if self.training:
            out_cls_list, out_c, out = self.decoder_G(compressed_y_G, nums_list, ground_truth_list, self.training)
        else:
            out_cls_list, out_c, out = self.decoder_G(compressed_y_G, nums_list=nums_list, ground_truth_list=[None] * 3,
                                                      training=self.training)
        torch.cuda.empty_cache()
        gc.collect()
        outA_gt = self.get_rgb(out, color)
        return [out2, out1, out0], None, compressed_y_G, out_cls_list, ground_truth_list, outA_gt
        # return [out2, out1, out0], None, compressed_y_G, out_cls_list, ground_truth_list

    def get_rgb(self, out, color):
        part1_in=time.time()
        feats = torch.zeros((len(out), 3))
        out_G = ME.SparseTensor(features=feats.float(), coordinates=out.C, coordinate_manager=color.coordinate_manager,
                                device=out.device)
        U = self.union(out_G, color)
        mask = isin(U.C, out_G.C)
        rec_coor = self.pruning(U, mask.to(U.device))
        mask = isin(U.C, color.C)
        mask = ~mask
        A_hat = self.pruning(U, mask.to(U.device))

        part1_time=time.time()-part1_in
        A_hat, color_time, tensor_time = rgb_get(A_hat, color,coordinate_manager=rec_coor.coordinate_manager)
        part2_in = time.time()
        final = self.union(A_hat, rec_coor)
        outA_gt = ME.SparseTensor(features=final.F.float(), coordinates=(final.C), device=A_hat.device)
        other_time=time.time()-part2_in+part1_time
        # return outA_gt, color_time, tensor_time, other_time
        # return outA_gt, color_time, tensor_time
        return outA_gt

    def save_residual_visualization(self, residual_st, file_path,
                                    mode="heatmap", cmap_name="hot", threshold=0.05):
        """
        Args:
            residual_st: MinkowskiEngine SparseTensor
            file_path: 输出 PLY 文件路径
            mode: 可视化模式
                  - "heatmap": 用 colormap 映射残差大小
                  - "threshold": 高亮超过阈值的点
            cmap_name: matplotlib colormap 名称 (默认 "hot")，仅在 heatmap 模式有效
            threshold: 残差阈值 (默认 0.05)，仅在 threshold 模式有效
        """
        # 1. 构造点云
        rec_pcd = open3d.geometry.PointCloud()
        rec_pcd.points = open3d.utility.Vector3dVector((residual_st.C[:, 1:].cpu()).float())

        # 2. 计算残差强度 (绝对值)
        F_abs = torch.abs(residual_st.F).cpu().numpy()  # [N, C]
        if F_abs.shape[1] > 1:  # 多通道，取均值
            F_val = F_abs.mean(axis=1)
        else:
            F_val = F_abs.squeeze()

        # 3. 可视化模式
        if mode == "heatmap":
            # 归一化到 [0,1]
            F_norm = (F_val - F_val.min()) / (F_val.max() - F_val.min() + 1e-8)
            cmap = cm.get_cmap(cmap_name)
            mapped_colors = cmap(F_norm)[:, :3]

        elif mode == "threshold":
            mapped_colors = np.zeros((len(F_val), 3))  # 初始化为黑色
            mask = F_val > threshold
            mapped_colors[mask] = [1, 0, 0]  # 超过阈值 = 红色
            mapped_colors[~mask] = [0.5, 0.5, 0.5]  # 其余点 = 灰色

        else:
            raise ValueError(f"Unknown mode: {mode}")

        rec_pcd.colors = open3d.utility.Vector3dVector(mapped_colors)

        open3d.io.write_point_cloud(file_path, rec_pcd, write_ascii=True)
        print(f"att点云已保存: {file_path} (mode={mode})")
    def code(self, x, color,filename):
        geo_enc=time.time()

        torch.cuda.empty_cache()
        gc.collect()
        out0 = ((self.conv0(x)))
        mask = self.score3_predictor_AtoG(out0)
        mask = F.softmax(mask.F, dim=-1)[:, 0]


        out0 = ME.SparseTensor(
            features=out0.F * mask.unsqueeze(1) + out0.F,
            coordinate_map_key=out0.coordinate_map_key,
            coordinate_manager=out0.coordinate_manager,
            device=out0.device)
        # self.save_residual_visualization(r_me, os.path.join(rootdir, filename + "_residual_me_heatmap.ply"),
        #                                  mode="heatmap", cmap_name="inferno")
        out0 = self.relu(self.down0(self.relu((out0))))
        out0 = self.block0(out0)
        mask = self.score4_predictor_AtoG(out0)
        mask = F.softmax(mask.F, dim=-1)[:, 0]
        out0 = ME.SparseTensor(
            features=out0.F * mask.unsqueeze(1) + out0.F,
            coordinate_map_key=out0.coordinate_map_key,
            coordinate_manager=out0.coordinate_manager,
            device=out0.device)

        # second scale
        out1 = self.relu(self.down1(self.relu(self.conv1(out0))))
        out1 = self.block1(out1)

        mask = self.score5_predictor_AtoG(out1)
        mask = F.softmax(mask.F, dim=-1)[:, 0]
        out1 = ME.SparseTensor(
            features=out1.F * mask.unsqueeze(1) + out1.F,
            coordinate_map_key=out1.coordinate_map_key,
            coordinate_manager=out1.coordinate_manager,
            device=out1.device)
        # third sclae
        out2 = self.relu(self.down2(self.relu(self.conv2(out1))))
        out2 = self.block2(out2)

        mask = self.score6_predictor_AtoG(out2)
        mask = F.softmax(mask.F, dim=-1)[:, 0]
        write_att = ME.SparseTensor(
            features=mask.unsqueeze(1) ,
            coordinate_map_key=out2.coordinate_map_key,
            coordinate_manager=out2.coordinate_manager,
            device=out2.device)
        # self.save_residual_visualization(write_att, os.path.join("vis_att", filename + "_att.ply"),
        #                                  mode="heatmap", cmap_name="inferno")
        # return None,None,None,None,None,None,None,None,None,None

        out2 = ME.SparseTensor(
            features=out2.F * mask.unsqueeze(1) + out2.F,
            coordinate_map_key=out2.coordinate_map_key,
            coordinate_manager=out2.coordinate_manager,
            device=out2.device)
        out2 = self.conv3(out2)
        geo_enc=time.time()-geo_enc

        rec_time=time.time()
        
        # A
        y_list = [out2, out1, out0]
        y_G = y_list[0]
        y_G_noise = torch.nn.init.uniform_(torch.zeros_like(y_G.F), -0.5, 0.5)
        if self.training:
            compressed_y_G = y_G.F + y_G_noise

        else:
            compressed_y_G = torch.round(y_G.F)
        compressed_y_G = ME.SparseTensor(
            features=compressed_y_G,
            coordinate_map_key=y_G.coordinate_map_key,
            coordinate_manager=y_G.coordinate_manager,
            device=y_G.device)
        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
                     for ground_truth in ground_truth_list]
        geo_enc=time.time()-geo_enc

        torch.cuda.empty_cache()
        gc.collect()


        if self.training:
            out_cls_list, out_c, out = self.decoder_G(compressed_y_G, nums_list, ground_truth_list, self.training)
        else:
            out_cls_list, out_c, out = self.decoder_G(compressed_y_G, nums_list=nums_list, ground_truth_list=[None] * 3,
                                                      training=self.training)
        torch.cuda.empty_cache()
        gc.collect()
        # geo_enc=time.time()-geo_enc
        # rec_time=time.time()
        # outA_gt, color_time, tensor_time, other_time = self.get_rgb(out, color)
        outA_gt, color_time, tensor_time=rgb_get(out,color)#traditional
        # outA_gt, color_time, tensor_time=rgb_get(out,color)#traditional
        rec_time=time.time()-rec_time


        # return [out2, out1, out0], None, compressed_y_G, out_cls_list, ground_truth_list, outA_gt,geo_enc, color_time, tensor_time, other_time, rec_time
        return [out2, out1, out0], None, compressed_y_G, out_cls_list, ground_truth_list, outA_gt,geo_enc, color_time, tensor_time, rec_time


class Decoder(torch.nn.Module):
    """the decoding network with upsampling.
    """

    def __init__(self, channels=[8, 64, 32, 16]):
        super().__init__()
        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels[1])

        self.conv0_cls = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up1 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels[2])

        self.conv1_cls = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block2 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels[3])

        self.conv2_cls = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        # for p in self.parameters():
        #     p.requires_grad=False
        # self.relu = ME.MinkowskiReLU(inplace=True)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training:
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else:
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned

    def forward(self, x, nums_list, ground_truth_list, training=True):
        #
        out = self.relu(self.conv0(self.relu(self.up0(x))))
        out = self.block0(out)
        out_cls_0 = self.conv0_cls(out)
        out = self.prune_voxel(out, out_cls_0,
                               nums_list[0], ground_truth_list[0], training)
        #
        out_c1 = out
        out = self.relu(self.conv1(self.relu(self.up1(out))))
        out = self.block1(out)
        out_cls_1 = self.conv1_cls(out)
        out = self.prune_voxel(out, out_cls_1,
                               nums_list[1], ground_truth_list[1], training)
        #
        out_c2 = out

        out = self.relu(self.conv2(self.relu(self.up2(out))))
        out = self.block2(out)
        out_cls_2 = self.conv2_cls(out)
        out = self.prune_voxel(out, out_cls_2,
                               nums_list[2], ground_truth_list[2], training)

        out_cls_list = [out_cls_0, out_cls_1, out_cls_2]
        out_c = [out_c1, out_c2, out]

        return out_cls_list, out_c, out
