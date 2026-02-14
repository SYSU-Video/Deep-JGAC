import os, sys, time, logging
from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME
from data_loader import yuv_rgb_tensor
from resscnn import ResSCNN
from loss import get_bce, get_bits, get_metrics,rgb_loss,rgb_loss_test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.tensorboard import SummaryWriter
import gc

class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)

        self.writer = SummaryWriter(log_dir=config.logdir)
        self.PCQA_loss=ResSCNN().to(device)
        self.model = model.to(device)
        self.optimizer = self.set_optimizer()
        self.logger.info(model)
        # self.lr_schedule=torch.optim.lr_scheduler.CyclicLR(self.optimizer,base_lr=self.config.lr*0.001,max_lr=self.config.lr,step_size_up=100,step_size_down=100,mode='triangular',scale_mode='cycle',cycle_momentum=False)
        # self.optimizer.load_state_dict(
        #     torch.load((self.dir + '/epoch_49.pth'), map_location=torch.device('cpu'))['optimizer_state_dict'])
        self.load_state_dict()
        self.epoch = 138
        self.record_set = {'bce':[],'mse':[], 'bces':[], 'bpp_A':[], 'bpp_G':[], 'bpp':[],'PCQA':[],'sum_loss':[], 'sum_loss_noPCQA':[]}
        self.loss=1e4
    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def load_state_dict(self):
        """selectively load model
        """
        loss_ckpts=torch.load('PCQM_Loss_checkpoints/test_checkpoint-LS.pth',map_location=torch.device('cpu'))
        self.PCQA_loss.load_state_dict(loss_ckpts['state_dict'])
        if self.config.init_ckpt=='':
            if self.config.rank==0:
                self.logger.info('Random initialization.')
        else:
            #ori load
            dir=self.config.init_ckpt
            self.dir=dir
            #
            #
            # ckpt_G = torch.load((dir + '/epoch_230.pth'), map_location=torch.device('cpu'))
            # ckpt_G = ckpt_G['model']
            # new_state_dict = self.model.state_dict()
            # # for name, param in new_state_dict.items():
            # #     new_state_dict[name] = param*0+0.0001
            #
            # print(1)
            # for name, param in ckpt_G.items():
            #     if name.startswith("encoder."):
            #         new_state_dict[name] = param
            #     if name.startswith("decoder."):
            #         new_name = "encoder.decoder_G" + name[len("decoder"):]
            #         new_state_dict[new_name] = param
            #         print(new_name)
            #
            #
            # ckpt_A = torch.load((dir+'/best_model.pth'),map_location=torch.device('cpu'))
            # ckpt_A=ckpt_A['model_state_dict']
            # ckpt_A_new = {}
            # for k, v in ckpt_A.items():
            #     name = k.replace('module.', '')  # remove `module.`
            #     ckpt_A_new[name] = v
            # for name, param in ckpt_A_new.items():
            #     if name.startswith("encoder."):
            #         if '_map' not in name:
            #             new_name = "encoder.A_" + name[len("encoder."):]
            #             new_state_dict[new_name] = param
            #             print(new_name)
            #     if name.startswith("decoder."):
            #         if '_map' not in name:
            #             new_name = "decoder_A" + name[len("decoder"):]
            #             new_state_dict[new_name] = param
            #             print(new_name)
            #     if name.startswith("hyer_encoder."):
            #         if '_map' not in name:
            #             new_name = "hyer_encoder_A" + name[len("hyer_encoder"):]
            #             new_state_dict[new_name] = param
            #             print(new_name)
            #     if name.startswith("hyer_decoder."):
            #         if '_map' not in name:
            #             new_name = "hyer_decoder_A" + name[len("hyer_decoder"):]
            #             new_state_dict[new_name] = param
            #             print(new_name)
            #     if name.startswith("bitEstimator_z."):
            #         if '_map' not in name:
            #             new_name = "bitEstimator_z_A" + name[len("bitEstimator_z"):]
            #             new_state_dict[new_name] = param
            #             print(new_name)
            #     if name.startswith("mu_transform."):
            #         if '_map' not in name:
            #             if name.endswith('.kernel')==False:
            #
            #                 new_name = "mu_transform_A" + name[len("mu_transform"):]
            #                 new_state_dict[new_name] = param
            #                 print(new_name)
            #     if name.startswith("scale_transform."):
            #         if '_map' not in name:
            #             if name.endswith('.kernel')==False:
            #
            #                 new_name = "scale_transform_A" + name[len("scale_transform"):]
            #                 new_state_dict[new_name] = param
            #                 print(new_name)
            #     if name.startswith("lrp."):
            #         if '_map' not in name:
            #             if name.endswith('.kernel')==False:
            #
            #                 new_name = "lrp_A" + name[len("lrp"):]
            #                 new_state_dict[new_name] = param
            #                 print(new_name)
            # ckpt_A.update(ckpt_G)
            # self.model.load_state_dict(ckpt_A,strict=False)
            # load_cpkt = self.model.load_state_dict(new_state_dict)


            #after train load
            new_state_dict = self.model.state_dict()

            new_state_dict_load=torch.load((dir + '/epoch_137.pth'), map_location=torch.device('cpu'))['model']
            for name, param in new_state_dict_load.items():
                if '_map' not in name:
                    # if '_GtoA' not in name:
                    new_state_dict[name] = param

            load_cpkt = self.model.load_state_dict(new_state_dict)
            # self.optimizer.load_state_dict(
            #     torch.load((self.dir + '/epoch_61.pth'), map_location=torch.device('cpu'))['optimizer_state_dict'])

            if self.config.rank==0:

                self.logger.info('Load checkpoint from ' + self.config.init_ckpt)


        return

    def save_model(self,best=None):
        if best == None:

            torch.save({'model': self.model.state_dict(),'optimizer_state_dict':self.optimizer.state_dict(),
                    'epoch': self.epoch,
                    },
            os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth'))
        if best==True:
            torch.save({'model': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': self.epoch,
                        },
                       os.path.join(self.config.ckptdir, 'best.pth'))
        return

    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items():
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items():
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))
        # return zero
        for k in self.record_set.keys():
            self.record_set[k] = []

        return

    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        if self.config.rank == 0:

            self.logger.info('Testing Files length:' + str(len(dataloader)))
        for batch_step, (coords, feats, color) in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            # data
            # if x.shape[0] > 6e5: continue
            # forward
            coords = coords.to(device)
            feats = feats.to(device)
            color = color.to(device)

            out_set = self.model(coords, feats, color, training=True)
            # loss
            bce, bce_list = 0, []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                curr_bce = get_bce(out_cls, ground_truth) / float(out_cls.__len__())
                # curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                bce += curr_bce
                bce_list.append(curr_bce.item())
            bpp = out_set['bpp']
            # metrics = []
            # mse_c=rgb_loss_test(out_set['out_A'],out_set['out_A_gt'])
            # print(mse_c)
            print('testinging')

            mse_c=rgb_loss(out_set['out_A'],out_set['out_A_gt'])
            PCQM_Loss=(5-self.PCQA_loss(out_set['out_A']))/5

            sum_loss = self.config.alpha * (bce )+ self.beta * bpp+self.config.lamda *mse_c+PCQM_Loss
            # for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
            #     metrics.append(get_metrics(out_cls, ground_truth))
            # record
            loss = sum_loss.cpu()

            if np.array(self.loss) > np.array(loss):
                self.loss = np.array(loss)
                self.save_model(best=True)
            if self.config.rank==0:
                self.record_set['bce'].append(bce.item())
                self.record_set['mse'].append((mse_c*self.config.lamda).item())
                self.record_set['bces'].append(bce_list)
                self.record_set['bpp_A'].append(out_set['bpp_A'].item())
                self.record_set['bpp_G'].append(out_set['bpp_G'].item())
                self.record_set['bpp'].append(bpp.item())
                self.record_set['PCQA'].append(PCQM_Loss.item())
                self.record_set['sum_loss'].append(sum_loss.item())
                self.record_set['sum_loss_noPCQA'].append((self.config.alpha * (bce )+ self.config.beta * bpp+self.config.lamda *mse_c).item())
                torch.cuda.empty_cache()# empty cache.
        if self.config.rank == 0 :

            self.record(main_tag=main_tag, global_step=self.epoch)

        return

    def train(self, dataloader):
        if self.config.rank == 0:

            self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        # optimizer
        # self.optimizer = self.set_optimizer()
        # self.optimizer.load_state_dict(
        #     torch.load((self.dir + '/epoch_49.pth'), map_location=torch.device('cpu'))['optimizer_state_dict'])

        if self.config.rank == 0:

            self.logger.info('alpha:' + str(round(self.config.alpha,2)) + '\tbeta:' + str(round(self.config.beta,2)))
            self.logger.info('LR:' + str(([params['lr'] for params in self.optimizer.param_groups])))
            self.logger.info('Training Files length:' + str(len(dataloader)))

            # dataloader

        start_time = time.time()
        for batch_step, (coords, feats,color) in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            # data
            # if x.shape[0] > 6e5: continue
            # forward
            coords=coords.to(device)
            feats=feats.to(device)
            color=color.to(device)

            a=self.model.encoder.conv4_AtoG._backward_hooks
            out_set = self.model(coords, feats,color)

            # loss
            bce, bce_list = 0, []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                curr_bce = get_bce(out_cls, ground_truth)/float(out_cls.__len__())
                # curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                bce += curr_bce
                bce_list.append(curr_bce.item())
            bpp = out_set['bpp']

            # if bpp>self.config.target_bpp:
            #     self.beta=1/(self.config.lamda_a)
            # else:
            self.beta =self.config.beta
            # mse_c = rgb_loss_test(out_set['out_A'], out_set['out_A_gt'])
            # print(mse_c)
            mse_c = rgb_loss(out_set['out_A'], out_set['out_A_gt'])
            # print(mse_c)
            ##PCQM_loss
            # pc_color=
            pc = ME.SparseTensor(features=(yuv_rgb_tensor(out_set['out_A'].F))-0.5, coordinates=out_set['out_A'].C, device=device)
            PCQM_Loss=self.PCQA_loss(pc)
            PCQM_Loss=(5-torch.mean(PCQM_Loss))/5
            sum_loss = self.config.alpha * (bce )+ self.beta * bpp+self.config.lamda *mse_c+PCQM_Loss
            # backward & optimize

            sum_loss.backward()
            self.optimizer.step()
            # self.lr_schedule.step()
            # metric & record
            with torch.no_grad():
                if self.config.rank == 0:
                    #
                    # metrics = []
                    # for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                    #     metrics.append(get_metrics(out_cls, ground_truth))
                    self.record_set['bce'].append(bce.item())
                    self.record_set['mse'].append((mse_c*self.config.lamda).item())
                    self.record_set['bces'].append(bce_list)
                    self.record_set['PCQA'].append(PCQM_Loss.item())
                    self.record_set['bpp_A'].append(out_set['bpp_A'].item())
                    self.record_set['bpp_G'].append(out_set['bpp_G'].item())
                    self.record_set['bpp'].append(bpp.item())
                    self.record_set['sum_loss'].append(sum_loss.item())
                    # loss=sum_loss.cpu()

                    # if np.array(self.loss) > np.array(loss):
                    #     print('well')
                    self.record_set['sum_loss_noPCQA'].append((self.config.alpha * (bce )+ self.beta * bpp+self.config.lamda *mse_c).item())
                    if (time.time() - start_time) > self.config.check_time*60:
                        self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
                        # self.save_model()
                        start_time = time.time()
            del out_set,mse_c,sum_loss,PCQM_Loss
            torch.cuda.empty_cache()# empty cache.
            gc.collect()
        with torch.no_grad():
            if self.config.rank==0:

                self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)

        if self.config.rank == 0:

            self.save_model()
            # if np.array(self.loss)>np.array(loss):
            #     self.loss=np.array(loss)
            #     self.save_model(best=True)
        torch.cuda.empty_cache()

        self.epoch += 1

        return
