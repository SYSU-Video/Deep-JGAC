import time, os, sys, glob, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,zz3'

import importlib
import numpy as np
import torch
import MinkowskiEngine as ME
from data_loader import PCDataset, make_data_loader,make_data_loader_mulgpu
from pcc_model import PCCModel
from trainer import Trainer
from torch.utils.data.distributed import DistributedSampler
##use python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py
from pathlib import Path
def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords, feats, color = list(zip(*list_data))
    coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)
    _, color = ME.utils.sparse_collate(coords, color)

    return coords_batch, feats_batch, color
def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--dataset", default='../../Downloads/数据集//RWTT_8i')
    parser.add_argument("--alpha", type=float, default=0.25, help="weights for xyz distoration.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")
    parser.add_argument("--lamda", type=float, default=400., help="weights for rgb distoration.")

    parser.add_argument("--lamda_a", type=float, default=0.1*(2e-6), help="control rate.")
    parser.add_argument("--lamda_b", type=float, default=0.1*2., help="control rate.")
    parser.add_argument("--target_bpp", type=float, default=0.8, help="control rate.")

    parser.add_argument("--init_ckpt", default='')#./ckpts
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument('--multigpu', default=False, type=bool)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--check_time", type=float, default=30,  help='frequency for recording state (min).')
    parser.add_argument("--prefix", type=str, default='tp', help="prefix of checkpoints/logger, etc.")
 
    args = parser.parse_args()

    return args

def traverse_path_recursively(rootdir):
        filedirs = []

        def gci(filepath):
            files = os.listdir(filepath)
            for fi in files:
                fi_d = os.path.join(filepath, fi)
                if os.path.isdir(fi_d):
                    gci(fi_d)
                else:
                    filedirs.append(os.path.join(fi_d))
            return

        gci(rootdir)

        return filedirs
class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, beta, lr, check_time,lamda,rank,target_bpp,lamda_a,lamda_b):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.beta = beta
        self.target_bpp = target_bpp
        self.lamda_a = lamda_a
        self.lamda_b = lamda_b
        self.lr = lr
        self.check_time=check_time
        self.lamda=lamda
        self.rank=rank
if __name__ == '__main__':
    # log
    args = parse_args()
    if args.multigpu:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
    device = torch.device('cuda', args.local_rank)
    experiment_dir = Path('./ckpts/'+str(args.alpha)+'_'+str(args.beta)+'_'+str(args.lamda))
    # experiment_dir.mkdir(exist_ok=True)
    training_config = TrainingConfig(
                            logdir=os.path.join('./logs', str(args.alpha)+'_'+str(args.beta)+'_'+str(args.lamda)),
                            ckptdir=experiment_dir,
                            init_ckpt=args.init_ckpt, 
                            alpha=args.alpha, 
                            beta=args.beta, 
                            lr=args.lr, 
                            check_time=args.check_time,
                            lamda=args.lamda,
                            rank=args.local_rank,
                            target_bpp=args.target_bpp,
                            lamda_a = args.lamda_a,
                            lamda_b=args.lamda_b)
    # model
    model = PCCModel()
    # trainer    
    trainer = Trainer(config=training_config, model=model)

    # dataset
    # filedirs = sorted(glob.glob(args.dataset+'*.ply'))
    filedirs_train = ('../../Downloads/数据集//RWTT_8i/')
    input_filedirs = traverse_path_recursively(rootdir=filedirs_train)
    filedirs_train = [f for f in input_filedirs if
                      (os.path.splitext(f)[1] == '.h5' or os.path.splitext(f)[1] == '.ply')]  # .off or .obj
    num=len(filedirs_train)
    trainer.logger.info('Training Files length:' + str((num)))
    train_dataset = PCDataset(filedirs_train)

    filedirs_val = sorted(glob.glob('8iVFB_val/'+ '*.ply'))

    test_dataset = PCDataset(filedirs_val)
    if args.multigpu:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = make_data_loader_mulgpu(dataset=train_dataset, train_sampler=train_sampler,batch_size=args.batch_size, shuffle=True,
                                           num_workers=8, repeat=False,
                                           collate_fn=collate_pointcloud_fn)
        val_sampler = DistributedSampler(test_dataset)

        test_dataloader = make_data_loader_mulgpu(dataset=test_dataset,train_sampler=val_sampler, batch_size=1, shuffle=True, num_workers=1,
                                         repeat=False,
                                         collate_fn=collate_pointcloud_fn)
    else:
        train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                            repeat=False)

        test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                           repeat=False)
    if args.multigpu:
        if args.local_rank == 0:

            print('multiple gpu used')
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)

    # training
    for epoch in range(0, args.epoch):
        if args.multigpu:
            train_sampler.set_epoch(epoch)
        if epoch>=50: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# update lr
        trainer.train(train_dataloader)
        if args.multigpu and args.local_rank==0:

            trainer.test(test_dataloader, 'Test')
