
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
 
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, \
                                          self.model.parameters()), self.args.lr, \
                                          momentum = self.args.momentum, \
                                          weight_decay = self.args.weight_decay)

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
        parser.add_argument('--base_dir', type=str)
        parser.add_argument('--score_save_dir', type=str, default="")
        parser.add_argument('--arch', type=str)
        parser.add_argument('--num_classes', type=int, default=7)
        parser.add_argument('--log', type=str)
        parser.add_argument('--pretrain', type=str)
        parser.add_argument('--resume', type=str)
        parser.add_argument('--model_dir', type=str)
        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--ckpt_save_freq', type=int, default=1)
        parser.add_argument('--test_freq', type=int, default=1)
        parser.add_argument('--emotions', type=str, default="Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise")
        parser.add_argument('--R', type=int, default=1)
        parser.add_argument('--only_test', action='store_true')

        args = parser.parse_args()
        args.steps = [int(x) for x in args.steps.split(',')] if args.steps else []
        args.emotions = args.emotions.split(',') if args.emotions else []

        return args
    

    def init_logger(self):
        
        logger = logging.getLogger("FER")
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
        train_set = getattr(data, self.args.train_dataset)(self.args.base_dir, self.args.emotions)
        test_set = getattr(data, self.args.test_dataset)(self.args.base_dir, self.args.emotions)
            
        train_loader = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    
        return train_loader, test_loader


    def init_model(self):
        best_acc = 0
        start_epoch = 1
        model = getattr(net, self.args.arch)( \
                pretrained=False, num_classes=self.args.num_classes, R=self.args.R) 

        if self.args.pretrain:
            self.logger.info("Loading pretrain: {}".format(self.args.pretrain))
            ckpt = torch.load(self.args.pretrain)
            model.load_state_dict(ckpt['state_dict'], strict = False)
            self.logger.info("Loaded pretrain: {}".format(self.args.pretrain))

        if self.args.resume:
            self.logger.info("Loading checkpoint: {}".format(self.args.resume))
            ckpt = torch.load(self.args.resume)
            start_epoch, best_acc = ckpt['epoch'], ckpt['best_acc']
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
        model_path = "{}/ckpt_{}.pth".format(self.args.model_dir, epoch_str)

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
        }, model_path)
        self.logger.info("Checkpoint saved to {}".format(model_path))
        return


    def train_epoch(self, epoch):
        self.model.train()
        t0 = time.time()
        cnt = 0
        acc_cnt = 0.
        lr = self.adjust_learning_rate(epoch)
        num_iter = len(self.train_loader)
    
        for iteration, batch in enumerate(self.train_loader):
            #torch.cuda.empty_cache()
            t_imgs = Variable(batch[0])
            t_labels = Variable(batch[1]).view(-1)

            if self.args.gpu >= 0:
                t_imgs = t_imgs.cuda()
                t_labels = t_labels.cuda()

            if self.args.R == 0 and len(t_imgs.shape) == 5:
                N, T, C, H, W = t_imgs.shape
                assert(T == 1)
                t_imgs = t_imgs.view(N, C, H, W)

            cnt += t_labels.size(0)
            self.optimizer.zero_grad()
            self.logger.debug("Shape: imgs={} labels={}".format( \
                              t_imgs.shape, t_labels.shape))
            
            output = self.model(t_imgs)
            acc_cnt += float(t_labels.eq(output.argmax(dim = 1)).sum())
            loss = self.criterion(output, t_labels)
            loss.backward()
            self.optimizer.step()

            acc = acc_cnt / cnt
            if iteration % self.args.print_freq ==  0:
                t1 = time.time()
                speed = (t1 - t0) / (iteration + 1)
                exp_time = self.format_second(speed * (num_iter * \
                          (self.args.epochs - epoch + 1) - iteration))

                self.logger.info("Epoch[{}/{}]({}/{}) Lr:{:.5f} Loss:{:.5f} " \
                                 "Acc:{:.5f} Speed:{:.2f} ms/iter {}" .format( \
                                 epoch, self.args.epochs, iteration, num_iter, \
                                 lr, loss.data, acc, speed * 1000, exp_time))
        return acc


    def test_once(self):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        self.model.eval()

        t0 = time.time()
        cnt = 0
        acc_cnt = 0.
        num_iter = len(self.test_loader)
        confuse_mat = np.zeros((self.args.num_classes, self.args.num_classes))

        with torch.no_grad():
            for iteration, batch in enumerate(self.test_loader):
                #torch.cuda.empty_cache()
                t_imgs = Variable(batch[0])
                t_labels = Variable(batch[1]).view(-1)

                if self.args.gpu >= 0:
                    t_imgs = t_imgs.cuda()
                    t_labels = t_labels.cuda()

                cnt += t_labels.size(0)
                
                N, frame_cnt, C, H, W = t_imgs.shape
                assert(N == 1)
                if self.args.R > 0:
                    output = self.model(t_imgs)
                else:
                    output = 0.
                    for fid in range(frame_cnt):
                        output += self.model(t_imgs[:, fid, :, :, :])

                # remove Neutral label
                pred_label = (output[:, 1:].argmax(dim = 1) + 1)[-1]
                if self.args.score_save_dir != "":
                    if not os.path.exists(self.args.score_save_dir):
                        os.makedirs(self.args.score_save_dir)
                    logits = output.softmax(dim = 1)
                    clip_id = batch[-1]['clip_id'][0]
                    with open(os.path.join(self.args.score_save_dir, clip_id + ".txt"), 'w') as ff:
                        for ind in range(frame_cnt):
                            ff.write("{0:04d}.jpg ".format(ind + 1))
                            ff.write("{:.0f}".format(int(t_labels)))
                            for item in list(logits[ind].cpu().numpy()):
                                ff.write(" {:.5f}".format(item))
                            ff.write("\n")
                
                acc_cnt += float(t_labels.eq(pred_label).sum())
                confuse_mat[t_labels, pred_label] += 1

                if iteration % 20 == 0:
                    self.logger.info("[{}/{}] Acc: {}/{}({:.3f})".format( \
                                     iteration, num_iter, int(acc_cnt), cnt, acc_cnt / cnt))

            t1 = time.time()
            speed = (t1 - t0) / num_iter * 1000

            # print the confusion matrix
            cstr = "\n     "
            emos = [x[:3] for x in self.args.emotions]
            for emotion in emos:
                cstr += emotion + "   "

            for idx, conf in enumerate(confuse_mat):
                cstr += "\n{:>3} ".format(emos[idx])
                for norm in conf / max(1., conf.sum()):
                    cstr += "{:.3f} ".format(norm)
                
                cstr += "\n{:>3} ".format(int(conf.sum()))
                for num in conf:
                    cstr += "{:>3}   ".format(int(num))

            self.logger.info(cstr)
            self.logger.info("Speed: {:.2f} ms/iter Acc: {}/{}({:.4f})".format(speed, \
                             int(acc_cnt), cnt, acc_cnt / cnt))

            torch.manual_seed(time.time())
            torch.cuda.manual_seed_all(time.time())

            return acc_cnt, cnt, acc_cnt / cnt


    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.args.only_test = False
            self.train_epoch(epoch)

            if epoch > 0 and epoch % self.args.ckpt_save_freq == 0:
                self.save_checkpoint(epoch)

            if epoch > 0 and epoch % self.args.test_freq == 0:
                self.args.only_test = True
                acc_cnt, cnt, acc = self.test_once()
                
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
