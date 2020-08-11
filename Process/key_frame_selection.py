import os
import cv2
import numpy as np
import json
from config import *


def parse_score_file(path):
    fin = open(path)
    lines = fin.readlines()
    cls = int(lines[0].split()[1])
    scores = np.array([[float(x) for x in line.split()[2:]] for line in lines])
    # smooth scores
    cnt = len(lines)
    step = cnt // 25
    res = np.array([scores[max(0, i-step):i+step+1, cls].mean() for i in range(cnt)])
    fin.close()
    return cls, res, scores[:, cls]


def save_smooth_score(src_dir, des_dir):
    fnames = os.listdir(src_dir)
    fnames.sort()
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        des = os.path.join(des_dir, fname)

        cls, res, _ = parse_score_file(src)
        cnt = res.shape[0]
        fout = open(des, 'w')
        for i in range(cnt):
            fout.write("{:0>4d}.jpg {}".format(i, cls))
            for ii in range(7):
                if ii == cls:
                    fout.write(" {:.5f}".format(res[i]))
                else:
                    fout.write(" 0.00000")
            fout.write('\n')
        fout.close()


def pre_intensity(score):
    if score < 0.167:
        return 0
    elif score < 0.5:
        return 1
    elif score < 0.833:
        return 2
    else:
        return 3


def get_key_frame(score_path, key_dir):
    clip_id = score_path.split('/')[-1].split('.')[0]
    cls, scores, origins = parse_score_file(score_path)
    cnt = scores.shape[0]
    
    # f''(x)
    d2 = [np.abs(scores[1] - scores[0])]
    for i in range(1, cnt - 1):
        d2.append(np.abs((scores[i + 1] + scores[i - 1]) - 2 * scores[i]))
    d2.append(np.abs(scores[cnt - 1] - scores[cnt - 2]))

    # group select index with max second derivative
    step = cnt // 15
    tmps = []
    for i in range(0, cnt, step):
        end = min(cnt, i + step)
        ind = np.array(d2[i:end]).argsort()[::-1][0] + i
        tmps.append(ind)

    tops = []
    i = 0
    while i < len(tmps):
        ind = tmps[i]
        # the first key frame should be the first frame
        # the last key frame should be the last frame
        if ind < step / 2 or ind >= cnt - step / 2:
            i += 1
            continue
        # the second last key frame
        if i + 1 >= len(tmps):
            tops.append(ind)
            i += 1
            continue
        # overlap
        ind2 = tmps[i + 1]
        if np.abs(ind2 - ind) < step / 2:
            valid_ind = ind if d2[ind] >= d2[ind2] else ind2
            tops.append(valid_ind)
            i += 2
            continue
        # no overlap
        tops.append(ind)
        i += 1

    # first and last frame as key frame
    if 0 not in tops:
        tops.append(0)
    if cnt - 1 not in tops:
        tops.append(cnt - 1)
    tops.sort()
    assert(0 == tops[0] and cnt - 1 == tops[len(tops) - 1])

    # remove continuous key frames with same intensity level
    ress = [0]
    for i in range(1, len(tops) - 1):
        inten1 = pre_intensity(scores[tops[i - 1]])
        inten2 = pre_intensity(scores[tops[i + 1]])
        inten = pre_intensity(scores[tops[i]])
        if inten == inten1 and inten == inten2:
            continue
        ress.append(tops[i])
    ress.append(cnt - 1)

    # write to the file
    des_f = os.path.join(key_dir, score_path.split('/')[-1])
    fout = open(des_f, 'w')
    for i, k in enumerate(ress):
        fout.write("{} {:0>4d}.jpg {} {:.5f}\n".format(k, k + 1, cls, scores[k]))
        
    fout.close()

    return len(ress)

    
if __name__ == "__main__":
    movies = Movies()
    emotions = movies.emotions()
    base_dir = movies.base_dir()

    dataset_dir = base_dir + "/dataset/"
    score_dir = base_dir + "/predict_score/"
    smooth_score_dir = base_dir + "/predict_score_smooth/"
    key_dir = base_dir + "/key_frames/"

    mkdir(smooth_score_dir)
    mkdir(key_dir)
    save_smooth_score(score_dir, smooth_score_dir)

    key_cnts = 0

    fnames = sort_listdir(score_dir)
    for idx, fname in enumerate(fnames):
        #print("[{}] {} key_cnts={}".format(idx, fname, key_cnts))
        key_cnt = get_key_frame(os.path.join(score_dir, fname), key_dir)
        key_cnts += key_cnt
    
    print("Samples:{} Key-Frames:{} Key-Frames-Each-Sample:{}".format(len(fnames), key_cnts, key_cnts / float(len(fnames))))
