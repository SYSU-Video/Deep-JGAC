import open3d as o3d
from data_loader import yuv_rgb
import numpy as np
def sp2ply(x_dec,recfile):
    # geo=x_dec.C
    # c=x_dec.F
    rec_pcd = o3d.geometry.PointCloud()  # 定义点云
    rec_pcd.points = o3d.utility.Vector3dVector((x_dec.C[:, 1:].cpu()).float())  # 定义点云坐标位置[N,3]
    rec_pcd.colors = o3d.utility.Vector3dVector(yuv_rgb((x_dec.F.cpu())))  # 定义点云坐标位置[N,3]
    o3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)
def split_batch_sp(sp_tensor):
    x_hat_G,x_hat_F=sp_tensor.decomposed_coordinates_and_features
    for i,j in enumerate(x_hat_G):
        tmp=np.array(j)
        

