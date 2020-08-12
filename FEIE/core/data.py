
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
from tools import *

from torchvision import transforms
from torch.utils.data import Dataset


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
    


# Facial Expression Intensity Dataset
class FEIDatasetTrain():
    # R: front and back continuous frame radius
    def __init__(self, emotion, fold, R, data_dir):
        self.samples = []
        seq = True if R > 0 else False
        self.transform = get_transform("train", seq)
        self.flips = [True, False]
        self.angles = [-4, -2, None, 2, 4]

        # split based on train&test, 6 emotions
        res = get_clip_dirs(data_dir)
        k_folds = res['split']
        assert(fold == 0)
        dirs = res['emotions'][emotion]
        dirs = list(filter(lambda x: k_folds[x.split('/')[-1]] == 'train', dirs))

        emotion_maps = {"Anger":1, "Disgust":2, "Fear":3, \
                        "Happiness":4, "Sadness":5, "Surprise":6}
        for dd in dirs:
            labels = read_intensity(os.path.join(dd, "annotation.xml"))
            frame_ids = sort_listdir(os.path.join(dd, "normalize_256"))

            frame_cnt = len(frame_ids)
            emotion_label = emotion_maps[dd.split('/')[-2]] 
            clip_id = dd.split('/')[-1]
            
            for i in range(0, frame_cnt):
                Ts = gen_conT(R, i, 0, frame_cnt)
                assert(len(Ts) == R * 2 + 1)
                assert(Ts[R] == i)

                face_paths = ["{}/normalize_256/{}".format(dd, frame_ids[x]) for x in Ts]
                intensities = [labels[x] for x in Ts]

                self.samples.append({"face_paths": face_paths,
                                     "intensities": intensities,
                                     "emotion_label": emotion_label,
                                     "clip_id": clip_id})
        
        logger = logging.getLogger("FEID")
        logger.info("FEIDTrainDataset-fold{} init: {}".format(fold, len(self.samples)))

    def __getitem__(self, index):
        sample = self.samples[index]
        flip = random.choice(self.flips)
        angle = random.choice(self.angles)

        # continuous 2R+1 frames
        t_imgs = []
        for face_path in sample['face_paths']:
            img = Image.open(face_path)
            img = img.rotate(angle) if angle else img
            img = img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img

            t_img = self.transform(img)
            C, H, W = t_img.shape
            t_imgs.append(t_img.view(1, C, H, W))
        t_imgs = torch.cat(t_imgs, dim = 0)

        # label info
        info = {"clip_id": sample['clip_id']}
        t_label = torch.LongTensor([sample['emotion_label']])
        t_intensities = torch.Tensor([sample['intensities']])
        #print("debug: info={} label={} intens={} shape={}".format(info, t_label, t_intensities, t_imgs.shape))
        return t_imgs, t_label, t_intensities, info

    def __len__(self):
        return len(self.samples)


# Facial Expression Intensity Dataset for Test
class FEIDatasetTest():
    # R: front and back continuous frame radius
    def __init__(self, emotion, fold, R, data_dir):
        self.samples = []
        self.transform = get_transform("test")

        # split based on train&test, 6 emotions
        res = get_clip_dirs(data_dir)
        k_folds = res['split']
        assert(fold == 0)
        dirs = res['emotions'][emotion]
        dirs = list(filter(lambda x: k_folds[x.split('/')[-1]] == 'test', dirs))
            
        emotion_maps = {"Anger":1, "Disgust":2, "Fear":3, \
                        "Happiness":4, "Sadness":5, "Surprise":6}
        for dd in dirs:
            labels = read_intensity(os.path.join(dd, "annotation.xml"))
            frame_ids = sort_listdir(os.path.join(dd, "normalize_256"))

            frame_cnt = len(frame_ids)
            emotion_label = emotion_maps[dd.split('/')[-2]] 
            clip_id = dd.split('/')[-1]
            
            face_paths = ["{}/normalize_256/{}".format(dd, frame_ids[x]) for x in range(frame_cnt)]
            self.samples.append({"face_paths": face_paths,
                                 "intensity": labels,
                                 "emotion_label": emotion_label,
                                 "clip_id": clip_id})
        
        logger = logging.getLogger("FEID")
        logger.info("FEIDTestDataset-fold{} init: {}".format(fold, len(self.samples)))

    def __getitem__(self, index):
        sample = self.samples[index]

        # continuous 2R+1 frames
        t_imgs = []
        for face_path in sample['face_paths']:
            t_img = self.transform(Image.open(face_path))
            C, H, W = t_img.shape
            t_imgs.append(t_img.view(1, C, H, W))
        t_imgs = torch.cat(t_imgs, dim = 0)

        # label info
        info = {"clip_id": sample['clip_id']}
        t_label = torch.LongTensor([sample['emotion_label']])
        t_intensity = torch.Tensor([sample['intensity']])
        #print("debug: info={} label={} intens={} shape={}".format(info, t_label, t_intensities, t_imgs.shape))
        return t_imgs, t_label, t_intensity, info

    def __len__(self):
        return len(self.samples)



# CK+ Intensity Dataset for Test
class CKPlusDatasetTest():
    # R: front and back continuous frame radius
    def __init__(self, emotion, fold, R):
        self.samples = []
        self.transform = get_transform("test")

        # split based on train&test, 6 emotions
        res = get_ckplus_clip_dirs()
        k_folds = res['split']
        dirs = res['emotions'][emotion]
        dirs = list(filter(lambda x: k_folds[x.split('/')[-1]] == fold, dirs))
            
        emotion_maps = {"Angry":1, "Contempt":2, "Disgust":3, "Fear":4, \
                        "Happy":5, "Sad":6, "Surprise":7}
        for dd in dirs:
            frame_ids = sort_listdir(os.path.join(dd, "normalize_256"))
            frame_cnt = len(frame_ids)
            labels = assign_intensity(frame_cnt)
            emotion_label = emotion_maps[dd.split('/')[-2]] 
            clip_id = dd.split('/')[-1]
            
            face_paths = ["{}/normalize_256/{}".format(dd, frame_ids[x]) for x in range(frame_cnt)]
            self.samples.append({"face_paths": face_paths,
                                 "intensity": labels,
                                 "emotion_label": emotion_label,
                                 "clip_id": clip_id})
        
        logger = logging.getLogger("FEID")
        logger.info("CKPlusTestDataset-fold{} init: {}".format(fold, len(self.samples)))

    def __getitem__(self, index):
        sample = self.samples[index]

        # continuous 2R+1 frames
        t_imgs = []
        for face_path in sample['face_paths']:
            t_img = self.transform(Image.open(face_path))
            C, H, W = t_img.shape
            t_imgs.append(t_img.view(1, C, H, W))
        t_imgs = torch.cat(t_imgs, dim = 0)

        # label info
        info = {"clip_id": sample['clip_id']}
        t_label = torch.LongTensor([sample['emotion_label']])
        t_intensity = torch.Tensor([sample['intensity']])
        #print("debug: info={} label={} intens={} shape={}".format(info, t_label, t_intensities, t_imgs.shape))
        return t_imgs, t_label, t_intensity, info

    def __len__(self):
        return len(self.samples)


# CK Extend Intensity Dataset
class CKPlusDatasetTrain():
    # R: front and back continuous frame radius
    def __init__(self, emotion, fold, R, data_dir):
        self.samples = []
        seq = True if R > 0 else False
        self.transform = get_transform("train", seq)
        self.flips = [True, False]
        self.angles = [-4, -2, None, 2, 4]

        # split based on train&test, 6 emotions
        res = get_ckplus_clip_dirs(data_dir)
        k_folds = res['split']
        dirs = res['emotions'][emotion]
        dirs = list(filter(lambda x: k_folds[x.split('/')[-1]] != fold, dirs))

        emotion_maps = {"Angry":1, "Contempt":2, "Disgust":3, "Fear":4, \
                        "Happy":5, "Sad":6, "Surprise":7}
        for dd in dirs:
            frame_ids = sort_listdir(os.path.join(dd, "normalize_256"))
            frame_cnt = len(frame_ids)
            labels = assign_intensity(frame_cnt)
            emotion_label = emotion_maps[dd.split('/')[-2]] 
            clip_id = dd.split('/')[-1]
            
            face_paths = []
            intensities = []
            for i in range(0, frame_cnt):
                Ts = gen_conT(R, i, 0, frame_cnt)
                assert(len(Ts) == R * 2 + 1)
                assert(Ts[R] == i)

                face_paths.extend(["{}/normalize_256/{}".format(dd, frame_ids[x]) for x in Ts])
                intensities.extend([labels[x] for x in Ts])

                self.samples.append({"face_paths": face_paths,
                                     "intensities": intensities,
                                     "emotion_label": emotion_label,
                                     "clip_id": clip_id})
        
        logger = logging.getLogger("FEID")
        logger.info("CKPlusTrainDataset-fold{} init: {}".format(fold, len(self.samples)))

    def __getitem__(self, index):
        sample = self.samples[index]
        flip = random.choice(self.flips)
        angle = random.choice(self.angles)

        # continuous 2R+1 frames
        t_imgs = []
        for face_path in sample['face_paths']:
            img = Image.open(face_path)
            img = img.rotate(angle) if angle else img
            img = img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img

            t_img = self.transform(img)
            C, H, W = t_img.shape
            t_imgs.append(t_img.view(1, C, H, W))
        t_imgs = torch.cat(t_imgs, dim = 0)

        # label info
        info = {"clip_id": sample['clip_id']}
        t_label = torch.LongTensor([sample['emotion_label']])
        t_intensities = torch.Tensor([sample['intensities']])
        #print("debug: info={} label={} intens={} shape={}".format(info, t_label, t_intensities, t_imgs.shape))
        return t_imgs, t_label, t_intensities, info

    def __len__(self):
        return len(self.samples)
