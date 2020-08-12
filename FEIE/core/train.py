
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

class Trainer(object):
    
    def __init__(self):
        self.args = self.init_args()
        self.logger = self.init_logger()        
        self.logger.info("Loading args: \n{}".format(self.args))

        if self.args.gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.args.gpu)
            if not torch.cuda.is_available():
                self.logger.error("No GPU={} found!".format(self.args.gpu))

        self.logger.info("Loading dataset:")
        self.train_loader, self.test_loader = self.init_dataset()

        self.logger.info("Loading model:")
        self.model, self.start_epoch, self.best_acc = self.init_model()
        self.model = self.model.cuda() if self.args.gpu >= 0 else self.model
 
        #self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        '''
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, \
                                          self.model.parameters()), self.args.lr, \
                                          momentum = self.args.momentum, \
                                          weight_decay = self.args.weight_decay)
        '''

    def init_args(self):
        
        parser = argparse.ArgumentParser()

        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--gpu', type=int, default=0, help="gpu id, < 0 means no gpu")
        parser.add_argument('--steps', type=str, default="", help='learning decay')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--train_dataset', type=str)
        parser.add_argument('--test_dataset', type=str)
        parser.add_argument('--dataset_dir', type=str)
        parser.add_argument('--arch', type=str)
        parser.add_argument('--log', type=str)
        parser.add_argument('--res', type=str)
        parser.add_argument('--pretrain', type=str)
        parser.add_argument('--resume', type=str)
        parser.add_argument('--model_dir', type=str)
        parser.add_argument('--emotion', type=str)
        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--ckpt_save_freq', type=int, default=1)
        parser.add_argument('--test_freq', type=int, default=1)
        parser.add_argument('--fold_index', type=int, default=0)
        parser.add_argument('--only_test', action='store_true')
        parser.add_argument('--R', type=int, default=0, help="Radius of continuous frames")
        parser.add_argument('--lstm_output', type=str, default="single", choices=['single', 'multi', 'weight_multi'])

        args = parser.parse_args()
        args.steps = [int(x) for x in args.steps.split(',')] if args.steps else []

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


    def init_dataset(self):
        train_set = getattr(data, self.args.train_dataset)(self.args.emotion, self.args.fold_index, self.args.R, self.args.dataset_dir)
        test_set = getattr(data, self.args.test_dataset)(self.args.emotion, self.args.fold_index, self.args.R, self.args.dataset_dir)
        
        train_loader = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    
        return train_loader, test_loader


    def init_model(self):
        best_acc = -1
        start_epoch = 1
        model = getattr(net, self.args.arch)( \
                pretrained=False, num_classes=1, R=self.args.R, lstm_output=self.args.lstm_output)

        if self.args.pretrain:
            self.logger.info("Loading pretrain: {}".format(self.args.pretrain))
            ckpt = torch.load(self.args.pretrain)
            model.load_state_dict(ckpt['state_dict'], strict = False)
            self.logger.info("Loaded pretrain: {}".format(self.args.pretrain))

        if self.args.resume:
            self.logger.info("Loading checkpoint: {}".format(self.args.resume))
            ckpt = torch.load(self.args.resume)
            start_epoch, best_acc = ckpt['epoch'], ckpt['best_acc']
            assert(ckpt['fold_index'] == self.args.fold_index)
            model.load_state_dict(ckpt['state_dict'], strict = True)
            self.logger.info("loaded checkpoint: {} (epoch {} best {:.2f})". \
                             format(self.args.resume, start_epoch, best_acc))

        return model, start_epoch, best_acc


    def adjust_learning_rate(self, epoch):
        ind = len(list(filter(lambda x: x <= epoch, self.args.steps)))
        lr = self.args.lr * (0.1 ** ind)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


    def format_second(self, secs):
        return "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format( \
               int(secs / 3600), int((secs % 3600) / 60), int(secs % 60))
    

    def save_checkpoint(self, epoch, best = False):
        epoch_str = "best" if best else "e{}".format(epoch)
        model_path = "{}/ckpt_fold{}_{}.pth".format(self.args.model_dir, self.args.fold_index, epoch_str)

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'fold_index': self.args.fold_index,
        }, model_path)
        self.logger.info("Checkpoint saved to {}".format(model_path))
        return


    def train_epoch(self, epoch):
        self.model.train()
        t0 = time.time()
        pred_intens = []
        gt_intens = []
        lr = self.adjust_learning_rate(epoch)
        num_iter = len(self.train_loader)
        print(num_iter)
    
        for iteration, batch in enumerate(self.train_loader):
            torch.cuda.empty_cache()
            N, PT, C, H, W = batch[0].shape
            T = self.args.R * 2 + 1
            assert(PT == T)
            
            t_imgs = Variable(batch[0])
            t_intensities = Variable(batch[2].view(-1, 1)) # N*2*T x 1  or N*T x 1
            t_single_intensity = Variable(batch[2].view(-1, T)[:, self.args.R].view(-1, 1)) # N*2 x 1 or N x 1
            #self.logger.info("Train intens-shape:{} imgs-shape:{} single_inten_shape:{}".format(t_intensities.shape, t_imgs.shape, t_single_intensity.shape))

            if self.args.gpu >= 0:
                t_imgs = t_imgs.cuda()
                t_intensities = t_intensities.cuda()
                t_single_intensity = t_single_intensity.cuda()

            self.optimizer.zero_grad()

            # static image input
            if self.args.R == 0:
                # N x 1
                output = self.model(t_imgs)
                pred = output * 1.
                loss = torch.sqrt((pred - t_single_intensity) ** 2 + 1e-12).mean()
            # only lstm
            else:
                if self.args.lstm_output == "multi":
                    output = self.model(t_imgs)
                    pred = output * 1.
                    loss = torch.sqrt((pred - t_intensities) ** 2 + 1e-12).mean()
                else:
                    output = self.model(t_imgs)
                    pred = output * 1.
                    loss = torch.sqrt((pred - t_single_intensity) ** 2 + 1e-12).mean()
                    
            loss.backward()
            self.optimizer.step()
            pred_intens.extend(pred.data.cpu().numpy()[:, 0].tolist())
            if self.args.lstm_output == "multi":
                gt_intens.extend(t_intensities.data.cpu().numpy()[:, 0].tolist())
            else:
                gt_intens.extend(t_single_intensity.data.cpu().numpy()[:, 0].tolist())

            acc = PCC(pred_intens, gt_intens)
            if iteration % self.args.print_freq ==  0:
                t1 = time.time()
                speed = (t1 - t0) / (iteration + 1)
                exp_time = self.format_second(speed * (num_iter * \
                          (self.args.epochs - epoch + 1) - iteration))

                self.logger.info("Epoch[{}/{}]({}/{}) Lr:{:.5f} Loss:{:.5f} " \
                                 "PCC:{:.5f} Speed:{:.2f} ms/iter {}" .format( \
                                 epoch, self.args.epochs, iteration, num_iter, \
                                 lr, loss.data, acc, speed * 1000, exp_time))
        return acc


    def test_once(self):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        self.model.eval()
        t0 = time.time()
        num_iter = len(self.test_loader)
        
        res_dir = os.path.dirname(self.args.res)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        fout = open(self.args.res, 'w')

        pccs = []
        iccs = []
        maes = []
        #labs = []

        with torch.no_grad():
            for iteration, batch in enumerate(self.test_loader):
                torch.cuda.empty_cache()
                
                N, frame_cnt, C, H, W = batch[0].shape
                T = 2 * self.args.R + 1
                assert(N == 1)

                t_imgs = Variable(batch[0])
                t_intensity = Variable(batch[2].view(-1)) # N
                info = batch[3]
                preds = []
                gts = t_intensity.data.cpu().numpy().tolist()

                if self.args.gpu >= 0:
                    #t_imgs = t_imgs.cuda()
                    t_intensity = t_intensity.cuda()

                # pred frame by frame
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

                #print("pred len:", len(preds), "gt len", len(gts))
                fout.write("\n")
                pcc = PCC(preds, gts)
                icc = ICC(preds, gts)
                mae = MAE(preds, gts)
                pccs.append(pcc)
                iccs.append(icc)
                maes.append(mae)
                                        
                #labs.extend(t_labels.data.cpu().numpy().tolist())
                if iteration % 5 == 0:
                    self.logger.info("[{}/{}] PCC: {:.5f} ICC:{:.5f} MAE:{:.5f}".format( \
                                     iteration, num_iter, np.array(pccs).mean(), \
                                     np.array(iccs).mean(), np.array(maes).mean()))
                
            t1 = time.time()
            speed = (t1 - t0) / num_iter * 1000

            # eval metric
            pcc = np.array(pccs).mean()
            icc = np.array(iccs).mean()
            mae = np.array(maes).mean()
            acc = pcc + icc - mae

            self.logger.info("For {} fold={}:".format(self.args.emotion, self.args.fold_index))
            self.logger.info("PCC: {:.5f}:".format(pcc))
            self.logger.info("ICC: {:.5f}:".format(icc))
            self.logger.info("MAE: {:.5f}:".format(mae))
            self.logger.info("Speed: {:.2f} ms/iter Avg: {:.5f}".format(speed, acc))
        
        torch.manual_seed(time.time())
        torch.cuda.manual_seed_all(time.time())

        return acc


    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.args.only_test = False
            self.train_epoch(epoch)

            if epoch > 0 and epoch % self.args.ckpt_save_freq == 0:
                self.save_checkpoint(epoch)

            if epoch > 0 and epoch % self.args.test_freq == 0:
                self.args.only_test = True
                acc = self.test_once()
                
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_checkpoint(epoch, best = True)
        return


    def run(self):
        if self.args.only_test:
            self.test_once()
        else:
            self.train()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
