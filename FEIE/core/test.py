
import os
import cv2
import sys
import math
import time
import torch
import logging
import argparse
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from metric import PCC, ICC, MAE

import net
import data

class Tester(object):
    
    def __init__(self):
        self.args = self.init_args()
        self.logger = self.init_logger()        
        self.logger.info("Loading args: \n{}".format(self.args))

        if self.args.gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.args.gpu)
            if not torch.cuda.is_available():
                self.logger.error("No GPU={} found!".format(self.args.gpu))

        model = getattr(net, self.args.arch)(pretrained=False, num_classes=1, R=self.args.R, lstm_output=self.args.lstm_output) 
        self.model = model.cuda() if self.args.gpu >= 0 else model


    def init_args(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=int, default=0, help="gpu id, < 0 means no gpu")
        parser.add_argument('--test_dataset', type=str)
        parser.add_argument('--dataset_dir', type=str)
        parser.add_argument('--arch', type=str)
        parser.add_argument('--log', type=str)
        parser.add_argument('--res', type=str)
        parser.add_argument('--k_fold', type=int)
        parser.add_argument('--resume', type=str)
        parser.add_argument('--emotion', type=str)
        parser.add_argument('--R', type=int, default=0, help="Radius of continuous frames")
        parser.add_argument('--lstm_output', type=str, default="single", choices=['single', 'multi', 'multi_weight', 'weight_multi'])

        args = parser.parse_args()
        args.resume = args.resume.split(',')
        assert(len(args.resume) == args.k_fold)
        return args
    

    def init_logger(self):

        if not os.path.exists("log"):
            os.makedirs("log")
        
        logger = logging.getLogger("FEID")
        logger.setLevel(level = logging.INFO)
        formatter = logging.Formatter("%(asctime)s-%(filename)s:%(lineno)d" \
                                      "-%(levelname)s-%(message)s")
    
        # log file stream
        handler = logging.FileHandler(self.args.log)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
    
        # log console stream
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
    
        logger.addHandler(handler)
        logger.addHandler(console)
    
        return logger


    def next_dataset(self, k_fold_index):
        test_set = getattr(data, self.args.test_dataset)(self.args.emotion, k_fold_index, self.args.R, self.args.dataset_dir)
        self.test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
        return


    def next_resume(self, k_fold_index):
        resume = self.args.resume[k_fold_index]
        ckpt = torch.load(resume)
        start_epoch, acc = ckpt['epoch'], ckpt['best_acc']
        assert(ckpt['fold_index'] == k_fold_index)
        self.model.load_state_dict(ckpt['state_dict'], strict = True)
        self.logger.info("loaded fold{} checkpoint: {} (epoch {} acc={})".\
                        format(k_fold_index, resume, start_epoch, acc))
        return


    def test_once(self, k_fold_index):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        self.model.eval()
        t0 = time.time()
        num_iter = len(self.test_loader)
        fout = open(self.args.res, 'a')
        pccs = []
        iccs = []
        maes = []
        #labs = []
        with torch.no_grad():
            for iteration, batch in enumerate(self.test_loader):
                #torch.cuda.empty_cache()
                info = batch[3]
                N, frame_cnt, C, H, W = batch[0].shape
                T = 2 * self.args.R + 1
                assert(N == 1)

                t_imgs = Variable(batch[0])
                t_intensity = Variable(batch[2].view(-1)) # N
                preds = []
                gts = t_intensity.data.cpu().numpy().tolist()
                if self.args.gpu >= 0:
                    #t_imgs = t_imgs.cuda()
                    t_intensity = t_intensity.cuda()

                fout.write('{}'.format(info['clip_id'][0]))
                for i in range(frame_cnt):
                    Ts = data.gen_conT(self.args.R, i, 0, frame_cnt)
                    input_imgs = [t_imgs[:, x, :, :, :].view(N, 1, C, H, W) for x in Ts]
                    input_imgs = torch.cat(input_imgs, dim = 1).cuda()

                    pred = self.model(input_imgs) * 1.
                    if self.args.lstm_output == "multi":
                        pred_val = pred.data.cpu().numpy().mean()
                        #pred_val = pred.data.cpu().numpy()[self.args.R, :].tolist()[0]
                    else:
                        pred_val = pred.data.cpu().numpy()[:, 0].tolist()[0]

                    preds.append(pred_val)
                    fout.write(' {:.5f}'.format(pred_val))
                fout.write("\n")
                pcc = PCC(preds, gts)
                icc = ICC(preds, gts)
                mae = MAE(preds, gts)
                pccs.append(pcc)
                iccs.append(icc)
                maes.append(mae)

                if iteration % 5 == 0:
                    self.logger.info("[{}/{}] PCC: {:.5f} ICC:{:.5f} MAE:{:.5f}".format( \
                                     iteration, num_iter, np.array(pccs).mean(), \
                                     np.array(iccs).mean(), np.array(maes).mean()))
                
                #self.logger.info("[{}/{}] PCC: {:.5f}".format( \
                #                     iteration, num_iter, PCC(gts, preds)))
            t1 = time.time()
            speed = (t1 - t0) / num_iter * 1000

            # eval metric
            pcc = np.array(pccs).mean()
            icc = np.array(iccs).mean()
            mae = np.array(maes).mean()
            acc = pcc + icc - mae

            self.logger.info("Emotion={} k_fold_index={}".format(self.args.emotion, k_fold_index))
            self.logger.info("PCC: {:.5f}:".format(pcc))
            self.logger.info("ICC: {:.5f}:".format(icc))
            self.logger.info("MAE: {:.5f}:".format(mae))
            self.logger.info("Speed: {:.2f} ms/iter Avg: {:.5f}".format(speed, acc))
        
        return pccs, iccs, maes


    def test(self):
        pccs, iccs, maes, = [], [], []
        f = open(self.args.res, 'w')
        f.close()
        for k in range(self.args.k_fold):
            self.next_dataset(k)
            self.next_resume(k)
            _pccs, _iccs, _maes = self.test_once(k)
            pccs.extend(_pccs)
            iccs.extend(_iccs)
            maes.extend(_maes)
        self.logger.info("----- Emotion={} K-fold Mean Metric-----".format(self.args.emotion))
        self.logger.info("++ PCC : {:.5f}".format(np.array(pccs).mean()))
        self.logger.info("++ ICC : {:.5f}".format(np.array(iccs).mean()))
        self.logger.info("++ MAE : {:.5f}".format(np.array(maes).mean()))
        return


if __name__ == "__main__":
    tester = Tester()
    tester.test()
