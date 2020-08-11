# -*- coding: UTF-8 -*-
import os
import numpy as np
from config import *


# return dict, key is clip_id, value is a list of txt paths
def tidy_labels_by_clip_id(label_dir):
    persons = os.listdir(label_dir)
    persons.remove('repeat')
    print("Labeled Persons: ", persons)
    res = {}
    cnt = 0
    for person in persons:
        cur_dir = os.path.join(label_dir, person)
        txts = os.listdir(cur_dir)
        for txt in txts:
            clip_id = txt.split('-')[2]
            if clip_id not in res.keys():
                res[clip_id] = []
            res[clip_id].append(os.path.join(cur_dir, txt))
            cnt += 1
    print("All label files: ", cnt)
    return res


def interpolate(path, frame_cnt, clip_id):
    f = open(path)
    items = f.readline().split()
    assert(int(items[2]) == 0)
    assert(int(items[-2]) == frame_cnt - 1)

    valid_inds = []
    # read the key frame labels
    key_cnt = int(items[1])
    keys = []
    for i in range(key_cnt):
        ind = int(items[2 + i * 2])
        val = float(items[3 + i * 2])
        val = 1. / 3 if val == 0.333 else val
        val = 2. / 3 if val == 0.667 else val
        keys.append((ind, val))
        valid_inds.append(ind)
    f.close()

    # calculate fixed gs
    prev_d = (keys[1][1] - keys[0][1]) / (keys[1][0] - keys[0][0])
    gs = [prev_d]
    for i in range(1, key_cnt - 1):
        cur_d = (keys[i+1][1] - keys[i][1]) / (keys[i+1][0] - keys[i][0])
        if cur_d * prev_d > 0:
            gs.append(0.5 * (cur_d + prev_d))
        else:
            gs.append(0)
        prev_d = cur_d
    gs.append(prev_d)

    # calculate the fk(x)
    intens = []
    prev_d = gs[0]
    x_prev = keys[0][0]
    y_prev = keys[0][1]
    for i in range(1, key_cnt):
        cur_d = gs[i]
        x = keys[i][0]
        y = keys[i][1]

        # ready for solve
        diff_x = x - x_prev
        diff_y = y - y_prev
        diff_g = cur_d - prev_d
        sum_x = x + x_prev
        sum_g = cur_d + prev_d
        x2 = x * x

        #f(x) = a*x^3+b*x^2+c*x^2+d
        #f(x_k) = y_k, f(x_k-1) = y_k-1, f'(x_k) = g_k, f'(x_k-1) = g_k-1
        a = (sum_g - 2 * (diff_y / diff_x)) / (diff_x * diff_x)
        b = 0.5 * (diff_g / diff_x - 3 * a * sum_x)
        c = cur_d - 2 * b * x - 3 * a * x2
        d = y - c * x - b * x2 - a * x * x2

        # get the intensity for middle frames
        for j in range(x_prev, x):
            j2 = j * j
            cur_y = a * j * j2 + b * j2 + c * j + d
            intens.append(min(max(cur_y, 0), 1))

        prev_d = cur_d
        x_prev = x
        y_prev = y

    #for frames after second-last key frames
    for j in range(x, frame_cnt):
        j2 = j * j
        cur_y = a * j * j2 + b * j2 + c * j + d
        intens.append(cur_y)

    assert(len(intens) == frame_cnt)
    # in case of all the label are same, pcc will be nan
    import random
    # in case of no any flucture
    if key_cnt == 2 and intens[0] == intens[-1]:
        for i in range(0, frame_cnt, 2):
            if intens[i] == 1:
                intens[i] -= 0.01
            elif intens[i] == 0:
                intens[i] += 0.01
            elif random.random() < 0.5:
                intens[i] += 0.01
            else:
                intens[i] -= 0.01
    return intens, valid_inds
        

def vote_label(paths, frame_cnt, clip_id):
    assert(len(paths) > 0)
    labels = []
    indss = []
    for path in paths:
        intens, inds = interpolate(path, frame_cnt, clip_id)
        labels.append(intens)
        indss.extend(inds)
    labels = np.array(labels)
    res = labels.mean(axis = 0)    
    return res, indss
        

def merge_persons(data_dir, emotions, inten_dir, label_dir):
    labels = tidy_labels_by_clip_id(label_dir)

    for emotion in emotions:
        print("For", emotion)
        base_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(base_dir):
            continue

        clip_ids = sort_listdir(base_dir)
        for clip_id in clip_ids:
            # merge all result that has been labeled
            frame_ids = sort_listdir(os.path.join(base_dir, clip_id + '/frames'))
            frame_cnt = len(frame_ids)
            frame_labels, key_inds = vote_label(labels[clip_id], frame_cnt, clip_id)

            # write the label result to the clip_dir
            clip_dir = os.path.join(base_dir, clip_id)
            fout = open(os.path.join(inten_dir, "{}.txt".format(clip_id)), 'w')
            for idx, frame_id in enumerate(frame_ids):
                is_key = 1 if idx in key_inds else 0 
                fout.write("{} {} {} {}\n".format(idx, frame_id, is_key, frame_labels[idx]))
            fout.close()


if __name__ == "__main__":
    movies = Movies()
    emotions = movies.emotions()
    base_dir = movies.base_dir()

    label_dir = base_dir + "/label/"
    dataset_dir = base_dir + "/dataset/"
    inten_dir = base_dir + "/intensity/"
    mkdir(inten_dir)

    merge_persons(dataset_dir, emotions, inten_dir, label_dir)
