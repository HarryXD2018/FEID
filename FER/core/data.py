
import os
import sys
import cv2
import copy
import math
import torch
import random
import logging
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


def sort_listdir(dd):
    xs = os.listdir(dd)
    xs.sort()
    return xs


def get_transform(phase, seq=False):
    assert(phase in ["train", "test"])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if phase == "test":
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize])
    else:
        if seq:
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        normalize])
        else:
            transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize])
    return transform
    

def collect_samples(base_dir, emotions):
    res = []
    for cls in range(0, len(emotions)):
        dd = os.path.join(base_dir, emotions[cls])
        if not os.path.exists(dd):
            continue
        clip_ids = sort_listdir(dd)
        for clip_id in clip_ids:
            if os.path.exists("{}/{}/normalize_256".format(dd, clip_id)):
                res.append({'clip_dir': os.path.join(dd, clip_id), 'class': cls})
    return res


def split_train_test(f):
    lines = open(f).readlines()
    d = {}
    for line in lines:
        items = line.split(" ")
        clip_id = items[0]
        d[clip_id] = items[1].rstrip()
    return d


class FEIDClsTrain():
    def __init__(self, base_dir, emotions):
        self.samples = []
        self.transform = get_transform('train', seq=True)
        smps = collect_samples(base_dir, emotions)

        k_folds = split_train_test(os.path.join(base_dir, "split_train_test.txt"))
        smps = list(filter(lambda x: k_folds[x['clip_dir'].split('/')[-1]] == 'train', smps))

        self.flips = [True, False]
        self.angles = [None]
        self.cnts = [12, 16, 20]

        for d in smps:
            frame_dir = os.path.join(d['clip_dir'], 'normalize_256')
            frame_ids = sort_listdir(frame_dir)
            frame_cnt = len(frame_ids)
            
            face_paths = [os.path.join(frame_dir, x) for x in frame_ids]
            self.samples.append({ "frame_cnt": frame_cnt,
                                  "face_paths": face_paths, 
                                  "class": d['class'],
                                  "clip_id": d['clip_dir'].split('/')[-1]})

        logger = logging.getLogger("FER")
        logger.info("FEID-Train init: clips: {}".format(len(smps)))

    def __getitem__(self, index):
        sample = self.samples[index]
        flip = random.choice(self.flips)
        angle = random.choice(self.angles)
        cnt = random.choice(self.cnts)

        step = float(sample['frame_cnt']) / cnt
        if step > 1:
            base_ind = random.choice(range(int(step)))
            Ts = [int(step * x + base_ind) for x in range(cnt)]
        else:
            Ts = list(range(sample['frame_cnt']))
        face_paths = [ sample['face_paths'][x] for x in Ts]

        # continuous 2R+1 frames
        t_imgs = []
        for face_path in face_paths:
            img = Image.open(face_path)
            img = img.rotate(angle) if angle else img
            img = img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img

            t_img = self.transform(img)
            C, H, W = t_img.shape
            t_imgs.append(t_img.view(1, C, H, W))
        t_imgs = torch.cat(t_imgs, dim = 0)
        info = {"clip_id": sample['clip_id']}
        t_label = torch.LongTensor([sample['class']])
        return t_imgs, t_label, info

    def __len__(self):
        return len(self.samples)


class FEIDClsTest():
    def __init__(self, base_dir, emotions):
        self.samples = []
        self.transform = get_transform('test')
        smps = collect_samples(base_dir, emotions)

        k_folds = split_train_test(os.path.join(base_dir, "split_train_test.txt"))
        smps = list(filter(lambda x: k_folds[x['clip_dir'].split('/')[-1]] == 'test', smps))

        for d in smps:
            frame_dir = os.path.join(d['clip_dir'], 'normalize_256')
            frame_ids = sort_listdir(frame_dir)
            frame_cnt = len(frame_ids)
            
            face_paths = [os.path.join(frame_dir, frame_ids[x]) for x in range(frame_cnt)]
            self.samples.append({"face_paths": face_paths,
                                 "class": d['class'],
                                 "clip_id": d['clip_dir'].split('/')[-1]})
        logger = logging.getLogger("FER")
        logger.info("FEIDCls-Test init: {}".format(len(self.samples)))

    def __getitem__(self, index):
        sample = self.samples[index]
        # continuous all frames in a clip
        t_imgs = []
        for face_path in sample['face_paths']:
            t_img = self.transform(Image.open(face_path))
            C, H, W = t_img.shape
            t_imgs.append(t_img.view(1, C, H, W))
        t_imgs = torch.cat(t_imgs, dim = 0)
        info = {"clip_id": sample['clip_id']}
        t_label = torch.LongTensor([sample['class']])
        return t_imgs, t_label, info

    def __len__(self):
        return len(self.samples)
