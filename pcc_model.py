import time

import torch
import MinkowskiEngine as ME

from autoencoder import Encoder, Decoder
from entropy_model import EntropyBottleneck
from utils import *
from bitEstimator import BitEstimator
import math
import gc
import numpy as np
import torchac
import gc
import pickle
import os
from coder import *
from data_utils import array2vector_torch
from range_code import *
from models.PCAC import get_model
from sp2ply import *
# PMF_model=pmf_model()
# PMF_model=nn.DataParallel(PMF_model)
class PCCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder_A = enconder()
        # for p in self.parameters():
        #     p.requires_grad=False
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        # self.decoder_G = Decoder(channels=[8,64,32,16])
        # self.hyer_encoder = Hyper_encoder_G(8)
        # self.hyer_decoder = Hyper_decoder_G(8)
        self.bitEstimator_z_G = BitEstimator(8)
        self.A_model=get_model()
        #ATTR

        self.hyer_encoder_G = Hyper_encoder_G(8)
        self.hyer_decoder_G = Hyper_decoder_G(8)
        self.use_hyper = True
        self.num_slices = 8
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = ME.MinkowskiMaxPooling(kernel_size=1, stride=2, dimension=3)
        self.pool_T = ME.MinkowskiPoolingTranspose(kernel_size=1, stride=2, dimension=3)
    def encode(self, coords, feats, color, filename):
        # G
        tt=time.time()
        device = feats.device
        x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
        x_color = ME.SparseTensor(features=color.float(), coordinates=coords, device=device)
        # tt=time.time()

        # y_list, y_A, compressed_y_G, out_cls_list, ground_truth_list, gt_color,geo_enc, color_time, tensor_time, other_time, rec_time = self.encoder.code(x, x_color)
        y_list, y_A, compressed_y_G, out_cls_list, ground_truth_list, gt_color,geo_enc, color_time, tensor_time, rec_time = self.encoder.code(x, x_color,filename)
        # return None,None,None,None,None,None,None,None,None,None

        y_G = y_list[0]
        num_points = [len(ground_truth) for ground_truth in y_list[1:] + [x]]
        ttt=time.time()-tt-rec_time

        ttt2=time.time()

        z_G = self.hyer_encoder_G(y_G)
        # ttt=time.time()-tt-rec_time

        compressed_z_G = torch.round(z_G.F)

        compressed_z_G = ME.SparseTensor(
            features=compressed_z_G,
            coordinate_map_key=z_G.coordinate_map_key,
            coordinate_manager=z_G.coordinate_manager,
            device=z_G.device)
        ttt2=time.time() - ttt2

        hyer_G = self.hyer_decoder_G(compressed_z_G)

        wirte_time = time.time()

        # code G feature
        mu = hyer_G.F[:, 0:8]
        scale = hyer_G.F[:, 8:]
        scale = torch.exp(scale).clamp(1e-10, 1e10)

        compressed_y_G_F = compressed_y_G.F
        ymin, ymax = compressed_y_G_F.min(), compressed_y_G_F.max()
        symbols = torch.arange(ymin, ymax + 1).cuda()
        symbols = symbols.reshape(1, -1).repeat(8, 1).repeat(len(mu), 1, 1)
        compressed_y = (compressed_y_G_F - ymin).type(torch.int16).cpu()
        if len(mu.shape) == 2:
            mu = mu.unsqueeze(2)
            scale = scale.unsqueeze(2)

        # pmf
        pmf = 0.5 - 0.5 * (symbols + 0.5 - mu).sign() * torch.expm1(-(symbols + 0.5 - mu).abs() / scale) \
              - (0.5 - 0.5 * (symbols - 0.5 - mu).sign() * torch.expm1(-(symbols - 0.5 - mu).abs() / scale))
        torch.cuda.empty_cache()
        gc.collect()
        pmf = pmf / (pmf.sum(dim=-1, keepdim=True))  # dim=K de dim
        # get CDF
        cdf = (self._pmf_to_cdf(pmf))
        cdf = cdf.clamp(max=1.).cpu()
        rootdir = './output'
        # z_G, ind_z = sort_spare_tensor(z_G)

        ind_z = torch.argsort(array2vector_torch(z_G.C,
                                                        z_G.C.max() + 1))
        z_G = ME.SparseTensor(features=(z_G.F[ind_z]),
                               coordinates=z_G.C[ind_z],
                               tensor_stride=z_G.tensor_stride[0],
                               coordinate_manager=z_G.coordinate_manager,
                               device=z_G.device)
        compressed_z_G = torch.round(z_G.F)
        z_string, zmin, zmax, z_shape = self.compress(compressed_z_G, 'G')

        idx_y = torch.argsort(array2vector_torch(compressed_y_G.C,
                                                 compressed_y_G.C.max() + 1))
        strings = torchac.encode_float_cdf(cdf[idx_y], compressed_y[idx_y], check_input_bounds=True)
        file_strings_hyper = os.path.join(rootdir, filename + '_G_F.strings_hyper')
        file_strings = os.path.join(rootdir, filename + '_G_F.strings')
        file_num = filename + '_num_points.bin'
        file_num = os.path.join(rootdir, file_num)
        with open(file_num, 'wb') as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())
        with open(file_strings_hyper, 'wb') as f:
            f.write(np.array(z_shape, dtype=np.int16).tobytes())  # [batch size, length, width, height, channels]
            f.write(np.array((zmin.cpu(), zmax.cpu()), dtype=np.int8).tobytes())
            f.write(np.array((ymin.cpu(), ymax.cpu()), dtype=np.int8).tobytes())
            f.write(z_string)
        with open(file_strings, 'wb') as f:
            f.write(strings)


        # code G Coord
        # ttt=time.time()-tt-rec_time
        filename_all = rootdir + '/' + filename  # no ply
        coordinate_coder = CoordinateCoder(filename_all)
        C_name = coordinate_coder.encode((compressed_y_G.C // compressed_y_G.tensor_stride[0]).detach().cpu()[:, 1:])

        wirte_time = time.time() -wirte_time
        ttt2=ttt2+wirte_time

        bytes_strings_hyper_G_F = os.path.getsize(file_strings_hyper)
        bytes_strings_G_F = os.path.getsize(file_strings)  # C_name
        bytes_strings_C = os.path.getsize(C_name)
        bytes_strings_num = os.path.getsize(file_num)

        bytes_G = bytes_strings_hyper_G_F + bytes_strings_G_F + bytes_strings_C + bytes_strings_num

        torch.cuda.empty_cache()
        gc.collect()

        # A
        y = self.A_model.encoder(gt_color)
        z = self.A_model.hyer_encoder(y)
        compressed_z = torch.round(z.F)
        compressed_z = ME.SparseTensor(
            features=compressed_z,
            coordinate_map_key=z.coordinate_map_key,
            coordinate_manager=z.coordinate_manager,
            device=z.device)
        hyer = self.A_model.hyer_decoder(compressed_z)
        indices_sort = torch.argsort(array2vector_torch(hyer.C,
                                                        hyer.C.max() + 1))
        hyer = ME.SparseTensor(features=(hyer.F[indices_sort]),
                               coordinates=hyer.C[indices_sort],
                               tensor_stride=hyer.tensor_stride[0],
                               coordinate_manager=hyer.coordinate_manager,
                               device=hyer.device)
        xslice = (y.F)[indices_sort].chunk(self.num_slices, 1)
        compressed_y = []
        rootdir = './output'
        if not os.path.exists('./output'):
            os.makedirs('./output')
        file_strings = './output/' + filename + '_color_F.bin'

        y_hat_slices = []
        mu_all = []
        scale_all = []
        for index, y_slice in enumerate(xslice):
            support_slices = (y_hat_slices)  # tensor list

            mu_support = torch.cat([hyer.F[:, :128]] + support_slices, dim=1)
            scale_support = torch.cat([hyer.F[:, 128:]] + support_slices, dim=1)

            mu_support = ME.SparseTensor(features=mu_support,
                                         coordinate_map_key=hyer.coordinate_map_key,
                                         coordinate_manager=hyer.coordinate_manager,
                                         device=hyer.device)
            scale_support = ME.SparseTensor(features=scale_support,
                                            coordinate_map_key=hyer.coordinate_map_key,
                                            coordinate_manager=hyer.coordinate_manager,
                                            device=hyer.device)
            mu = self.A_model.mu_transform[index](mu_support).F
            mu_all.append(mu)
            scale = torch.exp(self.A_model.scale_transform[index](scale_support).F).clamp(1e-10, 1e10)
            scale_all.append(scale)
            y_hat_slice = ste_round(y_slice - mu) + mu
            compressed_y.append((ste_round(y_slice - mu)).clone())
            lrp_support = torch.cat([mu_support.F, y_hat_slice], dim=1)
            lrp_support = ME.SparseTensor(features=lrp_support,
                                          coordinate_map_key=hyer.coordinate_map_key,
                                          coordinate_manager=hyer.coordinate_manager,
                                          device=hyer.device)
            lrp = self.A_model.lrp[index](lrp_support).F
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        mu = mu_all
        scale = scale_all
        # ymin, ymax = compressed_y1.min(), compressed_y1.max()

        strings = []
        # for ii in range(len(mu_all)):
        #     compressed_y[ii]=compressed_y[ii]-mu[ii]
        compressed_y1 = torch.cat(compressed_y, dim=-1)
        flag = torch.zeros(128, dtype=torch.int32)
        for ch_idx in range(128):
            if torch.sum(abs(compressed_y1[:, ch_idx])) > 0:
                flag[ch_idx] = 1
        flag = np.array(flag)
        flag = np.packbits(np.reshape(flag, [8, 128 // 8]))

        # non_zeor_idx = torch.squeeze(torch.nonzero(flag == 1))
        # non_zeor_idx=[]
        ymin, ymax = compressed_y1.min(), compressed_y1.max()

        symbols = torch.arange(ymin, ymax + 1).cuda()
        symbols1 = symbols.reshape(1, -1).repeat(128 // 8, 1).repeat(len(mu[0]), 1, 1)

        for ii in range(len(mu_all)):
            strings = sequence_code(strings, mu[ii], scale[ii], symbols1, compressed_y[ii], ymin)
        torch.cuda.empty_cache()
        gc.collect()

        indices_sort = torch.argsort(array2vector_torch(z.C,
                                                        z.C.max() + 1))
        z = ME.SparseTensor(features=(z.F[indices_sort]),
                            coordinates=z.C[indices_sort],
                            tensor_stride=z.tensor_stride[0],
                            device=z.device)
        compressed_z = torch.round(z.F)
        z_string, zmin, zmax, z_shape = self.compress(compressed_z, 'A')
        file_strings_hyper = os.path.join(rootdir, filename + '_color_F.strings_hyper')
        file_strings_nonZore = os.path.join(rootdir, filename + '_color_F_nonZore.strings')
        with open(file_strings_nonZore, 'wb') as f:
            f.write(np.array(flag, dtype=np.int16).tobytes())
        with open(file_strings_hyper, 'wb') as f:
            f.write(np.array(z_shape, dtype=np.int16).tobytes())  # [batch size, length, width, height, channels]
            f.write(np.array((zmin.cpu(), zmax.cpu()), dtype=np.int8).tobytes())
            f.write(np.array((ymin.cpu(), ymax.cpu()), dtype=np.int16).tobytes())
            f.write(z_string)
        with open(file_strings, 'wb') as f:
            pickle.dump(strings, f)
            # for byte_data in strings:
            #     f.write(byte_data+b'$$')
        bytes_strings_hyper_color = os.path.getsize(file_strings_hyper)
        bytes_strings_nonZOre_color = os.path.getsize(file_strings_nonZore)

        bytes_strings_color = os.path.getsize(file_strings)
        bytes_color = bytes_strings_hyper_color + bytes_strings_color + bytes_strings_nonZOre_color
        bit = bytes_G + bytes_color
        print('Total file size (Bytes): {}'.format(bit))
        print('Color Strings (Bytes): {}'.format(bytes_color))
        print('Geo Strings hyper (Bytes): {}'.format(bytes_G))
        # return bytes_strings_C, bytes_strings_G_F, bytes_strings_hyper_G_F, bytes_strings_num, bytes_strings_color, bytes_strings_hyper_color,geo_enc, color_time, tensor_time,other_time
        # return bytes_strings_C, bytes_strings_G_F, bytes_strings_hyper_G_F, bytes_strings_num, bytes_strings_color, bytes_strings_hyper_color,ttt,ttt2,color_time, tensor_time,other_time
        return bytes_strings_C, bytes_strings_G_F, bytes_strings_hyper_G_F, bytes_strings_num, bytes_strings_color, bytes_strings_hyper_color,ttt, color_time, tensor_time,gt_color

    def compress(self, x, type='A'):
        values = x
        xshape = x.shape
        min_v, max_v = values.min(), values.max()
        symbols = torch.arange(min_v, max_v + 1).cuda()
        symbols = symbols.reshape(1, -1).repeat(values.shape[-1], 1)
        values_norm = (values - min_v).type(torch.int16)
        pmf = self._likelihood_z(symbols.T, type)
        pmf = pmf.T
        pmf = pmf / (pmf.sum(dim=-1, keepdim=True))  # dim=K de dim
        cdf = self._pmf_to_cdf(pmf)
        cdf = cdf.clamp(max=1.)
        cdf = cdf.repeat(values.shape[0], 1, 1).cpu()
        strings = torchac.encode_float_cdf(cdf, values_norm.cpu(), check_input_bounds=True)
        return strings, min_v, max_v, xshape

    def _likelihood_z(self, z, type):
        if type == 'A':
            likelihood = self.A_model.bitEstimator_z(z + 0.5) - self.A_model.bitEstimator_z(z - 0.5)
        if type == 'G':
            likelihood = self.bitEstimator_z_G(z + 0.5) - self.bitEstimator_z_G(z - 0.5)
        return likelihood

    def decompress(self, strings, min_v, max_v, shape, type):
        symbols = torch.arange(min_v, max_v + 1)
        symbols = torch.tensor(symbols.reshape(1, -1).repeat(shape[-1], 1)).cuda()
        pmf = self._likelihood_z(symbols.T, type)
        pmf = pmf.T
        pmf = pmf / (pmf.sum(dim=-1, keepdim=True))  # dim=K de dim
        cdf = self._pmf_to_cdf(pmf)
        cdf = cdf.clamp(max=1.)
        cdf = cdf.repeat(shape[0], 1, 1).cpu()
        values = torchac.decode_float_cdf(cdf, strings)
        values += min_v
        return values.float()

    def decode(self, filename):

        print('===== Read binary files =====')
        rootdir = 'output'

        # decod G
        # dec_time=time.time()
        dec_time2=time.time()

        filename_all = rootdir + '/' + filename  # no ply
        coordinate_coder = CoordinateCoder(filename_all)
        y_G = coordinate_coder.decode()
        feats = np.expand_dims(np.zeros(y_G.shape[0]), 1).astype('int32')
        feats = torch.tensor(feats)
        y_G = torch.tensor(y_G, dtype=torch.int32)
        y_G = torch.cat((feats, y_G), dim=-1)
        indices_sort = torch.argsort(array2vector_torch(y_G, y_G.max() + 1))
        y_G = y_G[indices_sort]
        compressed_y = ME.SparseTensor(features=feats.float(), coordinates=y_G * 8, tensor_stride=8, device=device)
        z_coor = self.pool(self.pool(compressed_y))
        file_strings = os.path.join(rootdir, filename + '_G_F.strings')
        file_strings_hyper = os.path.join(rootdir, filename + '_G_F.strings_hyper')
        with open(file_strings_hyper, 'rb') as f:
            z_shape = np.frombuffer(f.read(2 * 2), dtype=np.int16)
            z_min_v, z_max_v = np.frombuffer(f.read(2 * 1), dtype=np.int8)
            y_min_v, y_max_v = np.frombuffer(f.read(2 * 1), dtype=np.int8)
            z_strings = f.read()
        with open(file_strings, 'rb') as f:
            y_strings = f.read()

        z = self.decompress(z_strings, z_min_v, z_max_v, z_shape, 'G')
        indices_sort = torch.argsort(array2vector_torch(z_coor.C,
                                               z_coor.C.max() + 1))
        compressed_z = ME.SparseTensor(features=z,
                                       coordinates=z_coor.C[indices_sort],
                                       coordinate_manager=z_coor.coordinate_manager,
                                       tensor_stride=z_coor.tensor_stride[0],
                                       device=z_coor.device)
        hyer_G = self.hyer_decoder_G(compressed_z)

        mu = hyer_G.F[:, 0:8]
        scale = hyer_G.F[:, 8:]
        scale = torch.exp(scale).clamp(1e-10, 1e10)

        ymin, ymax = y_min_v, y_max_v
        symbols = torch.arange(ymin, ymax + 1).cuda()
        symbols = symbols.reshape(1, -1).repeat(8, 1).repeat(len(mu), 1, 1)
        if len(mu.shape) == 2:
            mu = mu.unsqueeze(2)
            scale = scale.unsqueeze(2)

        # pmf
        pmf = 0.5 - 0.5 * (symbols + 0.5 - mu).sign() * torch.expm1(-(symbols + 0.5 - mu).abs() / scale) \
              - (0.5 - 0.5 * (symbols - 0.5 - mu).sign() * torch.expm1(-(symbols - 0.5 - mu).abs() / scale))
        torch.cuda.empty_cache()
        gc.collect()
        pmf = pmf / (pmf.sum(dim=-1, keepdim=True))  # dim=K de dim
        # get CDF
        cdf = (self._pmf_to_cdf(pmf))
        cdf = cdf.clamp(max=1.).cpu()
        torch.cuda.empty_cache()
        gc.collect()
        # _, idx_y = sort_spare_tensor(hyer_G)
        idx_y = torch.argsort(array2vector_torch(hyer_G.C,
                                                 hyer_G.C.max() + 1))
        # z_G = ME.SparseTensor(features=(z_G.F[ind_z]),
        #                       coordinates=z_G.C[ind_z],
        #                       tensor_stride=z_G.tensor_stride[0],
        #                       coordinate_manager=z_G.coordinate_manager,
        #                       device=z_G.device)

        y_G_F = torchac.decode_float_cdf(cdf[idx_y], y_strings) + ymin
        y_G = ME.SparseTensor(features=y_G_F.float(), coordinates=y_G * 8, tensor_stride=8, device=device)

        # ME.SparseTensor(
        #     features=y_G_F.float(),
        #     coordinate_map_key=hyer_G.coordinate_map_key,
        #     coordinate_manager=hyer_G.coordinate_manager,
        #     device=device)
        # ME.SparseTensor(features=y_F, coordinates=y_C * 8,
        #                 tensor_stride=8, device=device)
        # out_cls_list,out_c, out = self.encoder.decoder_G(y_G, nums_list, ground_truth_list, self.training)
        file_num = filename + '_num_points.bin'
        file_num = os.path.join(rootdir, file_num)
        with open(file_num, 'rb') as fin:
            num_points = np.frombuffer(fin.read(4 * 3), dtype=np.int32).tolist()
            num_points[-1] = int(1 * num_points[-1])  # update
            num_points = [[num] for num in num_points]
        dec_time2=time.time() - dec_time2

        dec_time=time.time()
        _, _, out = self.encoder.decoder_G(y_G, nums_list=num_points, ground_truth_list=[None] * 3,
                                           training=self.training)
        dec_time=time.time()-dec_time
        # sp2ply(out,'so.ply')
        # feats = np.expand_dims(np.zeros(out.shape[0]), 3).astype('int32')
        # feats = torch.tensor(feats)

        feats = torch.ones((out.shape[0], 1))
        # indices_sort = np.argsort(array2vector(out.C.cpu(),
        #                                        out.C.cpu().max() + 1))
        out = ME.SparseTensor(features=feats.float(), coordinates=(out.C),
                              device=out.device)

        out_min = self.pool(self.pool(self.pool(self.pool(self.pool(out)))))
        torch.cuda.empty_cache()
        gc.collect()
        # decode A
        print('===== Read binary files =====')
        rootdir = 'output'
        file_strings = os.path.join(rootdir, filename + '_color_F.bin')
        file_strings_nonZore = os.path.join(rootdir, filename + '_color_F_nonZore.strings')

        file_strings_hyper = os.path.join(rootdir, filename + '_color_F.strings_hyper')
        with open(file_strings_nonZore, 'rb') as f:
            flag = np.frombuffer(f.read(), dtype=np.int16)
        flag = flag.astype(np.uint8)
        flag = np.unpackbits(flag)
        flag = flag.reshape(8, 128 // 8)
        flag = torch.tensor(flag).cuda()
        with open(file_strings_hyper, 'rb') as f:
            z_shape = np.frombuffer(f.read(2 * 2), dtype=np.int16)
            z_min_v, z_max_v = np.frombuffer(f.read(2 * 1), dtype=np.int8)
            y_min_v, y_max_v = np.frombuffer(f.read(2 * 2), dtype=np.int16)
            z_strings = f.read()
        with open(file_strings, 'rb') as f:
            y_strings = pickle.load(f)
            # content=f.read()
            # y_strings=content.split(b'$$')[:-1]


        z = self.decompress(z_strings, z_min_v, z_max_v, z_shape, 'A')
        z = torch.tensor(z)

        indices_sort = torch.argsort(array2vector_torch(out_min.C,
                                                        out_min.C.max() + 1))
        compressed_z = ME.SparseTensor(features=z,
                                       coordinates=out_min.C[indices_sort],
                                       coordinate_manager=out_min.coordinate_manager,
                                       tensor_stride=out_min.tensor_stride[0],
                                       device=device)
        hyer = self.A_model.hyer_decoder(compressed_z)

        indices_sort = torch.argsort(array2vector_torch(hyer.C,
                                                        hyer.C.max() + 1))
        hyer = ME.SparseTensor(features=(hyer.F[indices_sort]),
                               coordinates=hyer.C[indices_sort],
                               tensor_stride=hyer.tensor_stride[0],
                               coordinate_manager=hyer.coordinate_manager,

                               device=hyer.device)
        y_hat_slices = []
        # symbols = torch.arange(y_min_v, y_max_v + 1).cuda()
        # symbols = symbols.reshape(1, -1)
        # mu = 0
        # scale = 0.1

        # pmf = 0.5 - 0.5 * (symbols + 0.5 - mu).sign() * torch.expm1(-(symbols + 0.5 - mu).abs() / scale) \
        #       - (0.5 - 0.5 * (symbols - 0.5 - mu).sign() * torch.expm1(-(symbols - 0.5 - mu).abs() / scale))
        #
        # pmf = pmf / (pmf.sum(dim=-1, keepdim=True))  # dim=K de dim
        # # get CDF
        # cdf = (self._pmf_to_cdf_cpu(pmf)).clamp(max=1.)
        # cdf = cdf.repeat(128 // self.num_slices, 1).repeat((len(hyer)), 1, 1)
        torch.cuda.empty_cache()
        gc.collect()
        symbols = torch.arange(y_min_v, y_max_v + 1).cuda()
        symbols = symbols.reshape(1, -1).repeat(128 // self.num_slices, 1).repeat(len(hyer), 1, 1)
        for index in range(self.num_slices):
            support_slices = (y_hat_slices)  # tensor list

            mu_support = torch.cat([hyer.F[:, :128]] + support_slices, dim=1)
            scale_support = torch.cat([hyer.F[:, 128:]] + support_slices, dim=1)

            mu_support = ME.SparseTensor(features=mu_support,
                                         coordinate_map_key=hyer.coordinate_map_key,
                                         coordinate_manager=hyer.coordinate_manager,
                                         device=hyer.device)
            scale_support = ME.SparseTensor(features=scale_support,
                                            coordinate_map_key=hyer.coordinate_map_key,
                                            coordinate_manager=hyer.coordinate_manager,
                                            device=hyer.device)

            mu_all = self.A_model.mu_transform[index](mu_support).F
            scale_all = torch.exp(self.A_model.scale_transform[index](scale_support).F).clamp(1e-10, 1e10)
            # y_hat_slice = ((torchac.decode_float_cdf(cdf, y_strings[index])) + y_min_v).cuda().float()
            y_hat_slice = sequence_decode(y_strings[index], flag[index], scale_all, symbols, y_min_v) + mu_all
            lrp_support = torch.cat([mu_support.F, y_hat_slice], dim=1)
            lrp_support = ME.SparseTensor(features=lrp_support,
                                          coordinate_map_key=hyer.coordinate_map_key,
                                          coordinate_manager=hyer.coordinate_manager,
                                          device=hyer.device)
            lrp = self.A_model.lrp[index](lrp_support).F
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)
            torch.cuda.empty_cache()
            gc.collect()
        y_hat_slices = torch.cat(y_hat_slices, dim=1)
        y_hat_slices = ME.SparseTensor(features=y_hat_slices,
                                       coordinate_map_key=hyer.coordinate_map_key,
                                       coordinate_manager=hyer.coordinate_manager,
                                       device=hyer.device)
        rec_x = self.A_model.decoder(y_hat_slices)
        return rec_x,dec_time,dec_time2

    def _pmf_to_cdf(self, pmf):
        pmf = pmf.cumsum(dim=-1)
        if len(pmf.shape) == 3:
            pmf = torch.cat((torch.zeros_like(pmf[:, :, :1]), pmf), dim=-1)
        else:
            pmf = torch.cat((torch.zeros_like(pmf[:, :1]), pmf), dim=-1)

        return pmf

    def _pmf_to_cdf_cpu(self, pmf):
        pmf = pmf.cumsum(dim=-1).detach().cpu()
        if len(pmf.shape) == 3:
            pmf = torch.cat((torch.zeros_like(pmf[:, :, :1]), pmf), dim=-1)
        else:
            pmf = torch.cat((torch.zeros_like(pmf[:, :1]), pmf), dim=-1)

        return pmf


    def forward(self, coords, feats,color, training=False):
        device=feats.device
        x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
        x_color=ME.SparseTensor(features=color.float(), coordinates=coords, device=device)

        NUM=len(x)
        # Encoder
        y_list,y_A,compressed_y_G,out_cls_list,ground_truth_list,gt_color = self.encoder(x,x_color)
        y_G = y_list[0]
        z_G = self.hyer_encoder_G(y_G)
        z_G_noise = torch.nn.init.uniform_(torch.zeros_like(z_G.F), -0.5, 0.5)
        if self.training:
            compressed_z_G = z_G.F + z_G_noise

        else:
            compressed_z_G = torch.round(z_G.F)

        compressed_z_G = ME.SparseTensor(
            features=compressed_z_G,
            coordinate_map_key=z_G.coordinate_map_key,
            coordinate_manager=z_G.coordinate_manager,
            device=z_G.device)
        hyer_G = self.hyer_decoder_G(compressed_z_G)

        # Decoder
        # z_A = self.hyer_encoder_A(y_A)
        # z_noise_A = torch.nn.init.uniform_(torch.zeros_like(z_A.F), -0.5, 0.5)
        # if self.training:
        #     compressed_z_A = z_A.F + z_noise_A
        # else:
        #     compressed_z_A = torch.round(z_A.F)
        # compressed_z_A = ME.SparseTensor(
        #     features=compressed_z_A,
        #     coordinate_map_key=z_A.coordinate_map_key,
        #     coordinate_manager=z_A.coordinate_manager,
        #     device=z_A.device)
        #     # out_G1 = ((self.encoder.conv0(out_recG)))
        #     #
        #     # out_G2 = self.encoder.relu(self.encoder.down0(self.encoder.relu((out_G1))))
        #     # out_G2 = self.encoder.block0(out_G2)
        #     #
        #     # out_G3 = self.encoder.relu(self.encoder.down1(self.encoder.relu(self.encoder.conv1(out_G2))))
        #     # out_G3 = self.encoder.block1(out_G3)
        #     #
        #     # out_G4 = self.encoder.relu(self.encoder.down2(self.encoder.relu(self.encoder.conv2(out_G3))))
        #     # out_G4 = self.encoder.block2(out_G4)
        #     # y_G_rec = self.encoder.conv3(out_G4)
        # if self.use_hyper:
        #     y_hat_slices = []
        #     mu_all=[]
        #     scale_all=[]
        #     hyer = self.hyer_decoder_A(compressed_z_A)
        #     xslice = y_A.F.chunk(self.num_slices, 1)
        #     for index, y_slice in enumerate(xslice):
        #         support_slices = (y_hat_slices)  # tensor list
        #
        #         mu_support = torch.cat([hyer.F[:,:128]] + support_slices, dim=1)
        #         scale_support = torch.cat([hyer.F[:,128:]] + support_slices, dim=1)
        #
        #         mu_support = ME.SparseTensor(features=mu_support,
        #                                   coordinate_map_key=y_A.coordinate_map_key,
        #                                   coordinate_manager=y_A.coordinate_manager,
        #                                   device=y_A.device)
        #         scale_support = ME.SparseTensor(features=scale_support,
        #                                      coordinate_map_key=y_A.coordinate_map_key,
        #                                      coordinate_manager=y_A.coordinate_manager,
        #                                      device=y_A.device)
        #         mu = self.mu_transform_A[index](mu_support).F
        #         scale = self.scale_transform_A[index](scale_support).F
        #         mu_all.append(mu)
        #         scale_all.append(scale)
        #         y_hat_slice=ste_round(y_slice-mu)+mu
        #         lrp_support=torch.cat([mu_support.F,y_hat_slice],dim=1)
        #         lrp_support = ME.SparseTensor(features=lrp_support,
        #                                         coordinate_map_key=y_A.coordinate_map_key,
        #                                         coordinate_manager=y_A.coordinate_manager,
        #                                         device=y_A.device)
        #         lrp = self.lrp_A[index](lrp_support).F
        #         lrp =0.5*torch.tanh(lrp)
        #         y_hat_slice+=lrp
        #
        #         y_hat_slices.append(y_hat_slice)
        #     mu_all = torch.cat(mu_all, dim=1)
        #     scale_all = torch.cat(scale_all, dim=1)
        #     y_hat_slices = torch.cat(y_hat_slices, dim=1)
        #     y_hat_slices = ME.SparseTensor(features=y_hat_slices,
        #                                     coordinate_map_key=y_A.coordinate_map_key,
        #                                     coordinate_manager=y_A.coordinate_manager,
        #                                     device=y_A.device)
        # mapkey=y_G.coordinate_map_key

        # decoded y information

        torch.cuda.empty_cache()
        gc.collect()

        # attr_recon = self.decoder_A(y_hat_slices)
        # MAP=attr_recon.coordinate_manager
        # if self.training:
        #     y_noise_A=torch.nn.init.uniform_(torch.zeros_like(y_A.F),-0.5,0.5)
        #     compressed_y_A=y_A.F+y_noise_A
        # else:
        #     compressed_y_A=torch.round(y_A.F)

        def get_bit(feature_G, mu_G, scale_G, z_G):
            # sigma_A = torch.exp(scale_A).clamp(1e-10, 1e10)
            # gaussian_A = torch.distributions.laplace.Laplace(mu_A, sigma_A,validate_args=True)
            # probs_A = gaussian_A.cdf(feature_A + 0.5) - gaussian_A.cdf(feature_A - 0.5)
            # total_bits_A = torch.sum(torch.clamp(-1.0 * torch.log(probs_A + 1e-10) / math.log(2.0), 0, 50))
            # prob_z_A = self.bitEstimator_z_A(z_A + 0.5) - self.bitEstimator_z_A(z_A - 0.5)
            # total_bits_A = torch.sum(torch.clamp(-1.0 * torch.log(prob_z_A + 1e-10) / math.log(2.0), 0, 50)) + total_bits_A

            sigma_G = torch.exp(scale_G).clamp(1e-10, 1e10)
            gaussian_G = torch.distributions.laplace.Laplace(mu_G, sigma_G, validate_args=True)
            probs_G = gaussian_G.cdf(feature_G + 0.5) - gaussian_G.cdf(feature_G - 0.5)
            # torch.distributions.normal
            total_bits_G = torch.sum(torch.clamp(-1.0 * torch.log(probs_G + 1e-10) / math.log(2.0), 0, 50))
            prob_z_G = self.bitEstimator_z_G(z_G + 0.5) - self.bitEstimator_z_G(z_G - 0.5)
            total_bits_G = torch.sum(
                torch.clamp(-1.0 * torch.log(prob_z_G + 1e-10) / math.log(2.0), 0, 50)) + total_bits_G

            BOTH = total_bits_G
            return BOTH, total_bits_G

        _, bit_G = get_bit(compressed_y_G.F, hyer_G.F[:, 0:8], hyer_G.F[:, 8:], compressed_z_G.F)
        # bit_G=get_bit_G(compressed_y_G.F,hyer_G.F[:,0:8],hyer_G.F[:,8:],compressed_z_G.F,'G')/NUM
        bpp_A, attr_recon, MSE=self.A_model(gt_color)
        # x_color=self.pool(self.pool(self.pool(x_color)))
        #
        # x_color_T=self.pool_T(x_color)

        # for i in range(B):
        #     geo_to_color = {}
        #     pc_ori=nums_list[0][i]
        #     color=nums_list[1][i]
        #     for j in range(pc_ori.shape[0]):
        #         geo = tuple(pc_ori[j,:3].tolist())
        #         color_j = color[j,:]
        #         geo_to_color[geo] = color_j
        #     target_color = []
        #     pc=nums_list_rec[i]
        #     for point in pc:
        #         geo = tuple(point.tolist())
        #         color = geo_to_color.get(geo)
        #         if color is not None:
        #             target_color.append(color)
        #         else:
        #             distances = torch.norm( pc_ori[:, :3].float()-point[:3].float(),dim=1)
        #             nearest_ind = torch.argmin(distances)
        #             target_color.append(pc_ori[nearest_ind, 3:])
        #     target_color=torch.cat(target_color,dim=0)
        #     color_all.append(target_color)
        # color=torch.cat(color_all,dim=0)
        # all_colors=[]
        # for i in range(B):
        #     geo_to_color = {}
        #     pc_ori=nums_list[0][i]
        #     pc_rec=nums_list_rec[i]
        #     color=nums_list[1][i]
        #     dis=torch.norm(pc_ori.float()-pc_rec[:,None,:],dim=-1)
        #     nn_ind=torch.argmin(dis,dim=0)
        #     target_colors=color[nn_ind,:]
        #     all_colors.append(target_colors)
        # color=torch.cat(all_colors,dim=0)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            'out_cls_list': out_cls_list,
            'bpp_G': bit_G / NUM,
            'bpp_A': bpp_A,
            'bpp': bit_G / NUM+bpp_A,
            'out_A': attr_recon,
            'out_A_gt': gt_color,
            'MSE':MSE,
            'ground_truth_list': ground_truth_list}

if __name__ == '__main__':
    model = PCCModel()
    print(model)

