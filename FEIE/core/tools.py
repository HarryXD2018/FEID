import os
import numpy as np
from xml.dom.minidom import parse
from xml.dom import minidom


def mkdir(dd):
    if not os.path.exists(dd):
        os.makedirs(dd)


def sort_listdir(dd):
    xs = os.listdir(dd)
    xs.sort()
    return xs


def gen_conT(R, i, low, high):
    l = i - R * 2
    r = i + R * 2 + 1
    Ts = []
    for x in range(l, r, 2):
        x = low if x < low else x
        x = high - 1 if x >= high else x
        Ts.append(x)
    return Ts


def read_intensity(fname):
    intens = []
    with open(fname, 'r') as fh:
        doc = minidom.parse(fh)
        root = doc.documentElement
        faces = root.getElementsByTagName('Faces')[0].getElementsByTagName('Face')
        for face in faces:
            intens.append(float(face.getAttribute("intensity")))
    return intens

def assign_intensity(frame_cnt):
    labels = [0]
    step = 1.0 / (frame_cnt - 1)
    for i in range(frame_cnt - 2):
        labels.append(step * (i + 1))
    labels.append(1.0)
    assert(len(labels) == frame_cnt)
    return labels

# return dict with keys expression and all
def get_clip_dirs(base_dir):
    split_f = base_dir + "/split_train_test.txt"
    sub_dirs = ["_", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
    
    vid_d = {}
    with open(split_f) as f:
        for line in f.readlines():
            items = line.split()
            vid_d[items[0]] = items[1].rstrip()

    res = {"all": [], "emotions": {}, "split": {}}
    for cls in range(1, len(sub_dirs)):
        res["emotions"][sub_dirs[cls]] = []
        emotion_dir = os.path.join(base_dir, sub_dirs[cls])
        clip_ids = sort_listdir(emotion_dir)
        for clip_id in clip_ids:
            res["split"][clip_id] = vid_d[clip_id]
            clip_dir = os.path.join(emotion_dir, clip_id)
            res["all"].append(clip_dir)
            res["emotions"][sub_dirs[cls]].append(clip_dir)
    return res


# return dict with keys expression and all
def get_ckplus_clip_dirs(base_dir):
    split_f = base_dir + "/../split_5_fold.txt"
    sub_dirs = ["_", "Angry", "Contempt", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    
    vid_d = {}
    with open(split_f) as f:
        for line in f.readlines():
            items = line.split()
            vid_d[items[1]] = int(items[2].rstrip())

    res = {"all": [], "emotions": {}, "split": {}}
    for cls in range(1, len(sub_dirs)):
        res["emotions"][sub_dirs[cls]] = []
        emotion_dir = os.path.join(base_dir, sub_dirs[cls])
        clip_ids = sort_listdir(emotion_dir)
        for clip_id in clip_ids:
            res["split"][clip_id] = vid_d[clip_id]
            clip_dir = os.path.join(emotion_dir, clip_id)
            res["all"].append(clip_dir)
            res["emotions"][sub_dirs[cls]].append(clip_dir)
    return res
