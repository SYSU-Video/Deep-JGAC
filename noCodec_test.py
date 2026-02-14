import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import collections
import MinkowskiEngine as ME
import open3d

import numpy as np
import torch
from data_loader import *
import matplotlib.pyplot as plt
from utils1.pc_error_wrapper import pc_error
import time
import importlib
import sys
import gc
import argparse
import re
import glob
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.set_default_dtype(torch.float32)
# np.seterr(all='ignore')
# np.seterr(default='ignore')
device_o3d=o3d.core.Device("CPU:0")
dtype_o3d=o3d.core.float32
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('point_based_PCGC')
    parser.add_argument('--dataset_path', type=str, default='./')
    return parser.parse_args()
if __name__ == '__main__':


    args = parse_args()
    ckpt_of_different_rates = ['0.25_1.0_400.0','0.5_1.0_1000.0','1_1.0_4000.0','2_1.0_8000.0','3_1.0_16000.0'
                               ]
    # filedirs_train = sorted(glob.glob('/home/zpc/Downloads/数据集/RWTT/generate/test_DATA_SIAT/'+ '*.ply'))
    filedirs_train = sorted(glob.glob('/home/zpc/Documents/PCQA/MPED-main/HVS_task/normal_true_all/'+ '*.ply'))
    # filedirs_train = sorted(glob.glob('/home/zpc/Downloads/数据集/dataset4/' + '*.ply'))
    # filedirs_train = sorted(glob.glob('./ba/'+ '*.ply'))

    test_data = PCDataset(filedirs_train)
    test_loader = make_data_loader(dataset=test_data, batch_size=1, shuffle=False,
                                        repeat=False)

    idx=0
    idx_rate=0
    for exp_name in ckpt_of_different_rates:
        model_name = 'pcc_model'
        experiment_dir = 'ckpts/' + exp_name + '/BEST50.pth'
        MODEL = importlib.import_module(model_name)
        model = MODEL.PCCModel().cuda()  # .cuda
        checkpoint = torch.load(str(experiment_dir))
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['model'].items():
            if '_map' not in k:
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        outdir = './output'
        if not os.path.exists(outdir): os.makedirs(outdir)
        print('=' * 10, 'Test', '=' * 10)
        model.eval()

        for step, (coords, feats, color) in enumerate(test_loader):
            with torch.no_grad():
                coords = coords.to(device)
                feats = feats.to(device)
                color = color.to(device)
                filedir=test_data.files[step]
                filename = os.path.split(filedir)[-1].split('.')[0]
                # if torch.cuda.is_available():
                #     xyz = xyz.cuda()
                #     pc_data = pc_data.cuda()  # [B,N,3]
                # total_decoder_output.append(out_model['out_A'].F.cpu())
                # total_bpp += bpp
                points = coords
                if torch.max(points) < 1024 and torch.max(points) >= 512:
                    bit_depth = 10
                    peak_value = 1023
                elif torch.max(points) < 2048 and torch.max(points) >= 1024:
                    bit_depth = 11
                    peak_value = 2047
                elif torch.max(points) < 4096 and torch.max(points) >= 2048:
                    bit_depth = 12
                    peak_value = 4095

                print(filename)
                # encode
                start_time = time.time()
                # out = model.encode(coords, feats, color,filename)
                out_set = model(coords, feats, color)

                enc_time=round(time.time() - start_time, 3)

                print('Enc Time:\t',enc_time, 's')
                bit=0
                bpp_G=out_set['bpp_G'].cpu()
                bpp_A=out_set['bpp_A'].cpu()
                bpp = out_set['bpp'].cpu()
                # bpp=8*(bit)
                # torch.cuda.empty_cache()
                # gc.collect()
                # model = MODEL.PCCModel().cuda()  # .cuda
                # model.eval()
                #
                # new_state_dict = collections.OrderedDict()
                # for k, v in checkpoint['model'].items():
                #     if '_map' not in k:
                #         name = k.replace('module.', '')  # remove `module.`
                #         new_state_dict[name] = v
                # model.load_state_dict(new_state_dict)
                # decode
                start_time = time.time()
                x_dec = out_set['out_A']
                dec_time=round(time.time() - start_time, 3)
                print('Dec Time:\t', dec_time, 's')

                # ori_pcd = open3d.geometry.PointCloud()  # 定义点云
                # ori_pcd.points = open3d.utility.Vector3dVector((coords[:, 1:].cpu()).float())  # 定义点云坐标位置[N,3]
                # ori_pcd.colors = open3d.utility.Vector3dVector(yuv_rgb((color.cpu())))  # 定义点云坐标位置[N,3]
                rec_pcd = open3d.geometry.PointCloud()  # 定义点云
                rec_pcd.points = open3d.utility.Vector3dVector((x_dec.C[:, 1:].cpu()).float())  # 定义点云坐标位置[N,3]
                rec_pcd.colors = open3d.utility.Vector3dVector(yuv_rgb((x_dec.F.cpu())))  # 定义点云坐标位置[N,3]
                if not os.path.exists('pc_file'): os.makedirs('pc_file')
                # orifile='pc_file/'+filename+'_ori.ply'
                recfile='pc_file/'+filename+'_r'+str(idx_rate)+'_rec.ply'
                open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)
                # open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
                # ori = o3d.t.io.read_point_cloud(orifile)
                # a=np.array((coords[:, 1:].cpu()).float())
                # ori.point.positions=o3d.core.Tensor(a,dtype_o3d,device_o3d)
                # o3d.t.io.write_point_cloud(orifile, ori, write_ascii=True)
                ori = o3d.t.io.read_point_cloud(recfile)
                a=np.array((x_dec.C[:, 1:].cpu()).float())
                ori.point.positions=o3d.core.Tensor(a,dtype_o3d,device_o3d)
                o3d.t.io.write_point_cloud(recfile, ori, write_ascii=True)
                pc_error_metrics = pc_error(infile1=filedir, infile2=recfile, res=peak_value)  # res为数据峰谷差值
                pc_errors = [pc_error_metrics["c[0],PSNRF"][0],
                             pc_error_metrics["c[1],PSNRF"][0],
                             pc_error_metrics["c[2],PSNRF"][0]]
                print(pc_errors)

                results = pc_error_metrics
                results["bpp_A"] = np.array(bpp_A).astype('float32')
                results["bpp_G"] = np.array(bpp_G).astype('float32')
                results["bpp"] = np.array(bpp).astype('float32')
                results["enc_time"] = np.array((enc_time)).astype('float32')
                results["dec_time"] = np.array(dec_time)
                a=os.path.split(str(test_data.files[step]))[-1].split('.')[0]
                a=re.split(r'\d+',a)[0]
                results['sequence'] = a
                last_col = results.pop(results.columns[-1])
                results.insert(0, last_col.name, last_col)
                if step == 0:
                    # global all_results
                    all_result = results.copy(deep=True)
                    # idx=1
                else:
                    all_result = all_result._append(results, ignore_index=True)
                    print('hello')
                del color,coords,points,x_dec
                torch.cuda.empty_cache()
                gc.collect()
                print(all_result)
        all_result=all_result.groupby('sequence').mean().reset_index()
        idx_rate+=1
        if idx == 0:
            global all_results
            all_results = all_result.copy(deep=True)
            idx = 1
        else:
            all_results = all_results._append(all_result, ignore_index=True)
            print('hello')
    if not os.path.exists('result'): os.makedirs('result')
    csv_name = os.path.join('./result/', 'buste.csv')
    all_results.to_csv(csv_name, index=False)




