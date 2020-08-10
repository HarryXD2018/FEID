import os
from config import *
import imageio
from face.src.detector_cuda import detect_faces, init_model
from PIL import Image
import math
import numpy as np


# get frames cnt and duration of a movie
def get_length_of_movie(movie_name, movie_path, base_dir):
    save_file = base_dir + "video_length.txt"

    if os.path.exists(save_file):
        lines = open(save_file).readlines()
        lines = list(filter(lambda x: x.split(' ')[0] == movie_name, lines))
        if len(lines) > 0:
            items = lines[0].split(' ')
            return float(items[1]), int(items[2])
    
    vid = imageio.get_reader(movie_path, 'ffmpeg')
    duration = float(vid.get_meta_data()['duration'])
    frame_count = int(vid.count_frames())
    with open(save_file, 'a') as f:
        f.write("{} {} {}\n".format(movie_name, duration, frame_count))

    return duration, frame_count


def format_t(secs):
    h = int(secs / 3600)
    m = int((secs % 3600) / 60)
    s = secs % 60
    micros = int((s - int(s)) * 100)
    ss = "{:0>2}:{:0>2}:{:0>2}.{:0>2}".format(h,m,int(s), micros)
    return ss

def cal_iou(rect1, rect2):
    cw1 = max(rect1[0], rect2[0])
    ch1 = max(rect1[1], rect2[1])
    cw2 = min(rect1[2], rect2[2])
    ch2 = min(rect1[3], rect2[3])

    if cw1 >= cw2 or ch1 >= ch2:
        return 0.
    
    inter = (cw2 - cw1) * (ch2 - ch1)
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    
    return float(inter) / (area1 + area2 - inter)


def init_emotion_model():
    return None


# detect faces and return, input img is pil image with channel RGB
def get_face(img, face_model):
    w, h = img.size
    ratio = 960.0 / float(w)
    if ratio < 1:
        nw, nh = int(w * ratio), int(h * ratio)
        img = img.resize((nw, nh), Image.BILINEAR)
    rects, landmarks = detect_faces(img, face_model)
    #rects, landmarks = detect_faces(img)
    if rects is not None and len(rects) > 0:
        w1, h1, w2, h2, score = rects[0]
        face_rect = [max(0, w1), max(0, h1), w2, h2]
        landmark = landmarks[0].astype(np.float32).reshape(-1, 5).transpose(1, 0)
        if ratio < 1:
             face_rect = [x / ratio for x in face_rect] 
             for pts in landmark:
                 pts[0], pts[1] = pts[0] / ratio, pts[1] / ratio

        return face_rect, landmark
    else:
        return None, None
    

# predict emtion by model, input PIL image
def emotion_dectect(imgs, faces, model):
    res = [{'index': 1, 'conf': 0.5, 'label':1},]
    return res

# write clips time, start-idx, end-idx, emotion label, frames, clips
def write_to_file(vid_path, sample_cnt, frames, faces, sidx, eidx, \
                  emotions, base_dir, frames_cnt, duration, lands):
    vid_id = vid_path.split('/')[-1].split('.')[0]
    clip_name = "{}_{:04d}".format(vid_id, sample_cnt)
    # save frames and faces
    frame_dir = os.path.join(base_dir, "frames/{}/{}".format(vid_id, clip_name))
    mkdir(frame_dir)
    fout = open(os.path.join(base_dir, "faces/" + clip_name + ".txt"), 'w')
    for idx, frame in enumerate(frames):
        frame_id = "{0:04d}.jpg".format(idx + 1)
        frame.save("{}/{}".format(frame_dir, frame_id))
        w1, h1, w2, h2 = faces[idx]
        fout.write("{} {} {} {} {}".format(frame_id, w1, h1, w2, h2))
        for pts in lands[idx]:
            fout.write(" {} {}".format(pts[0], pts[1]))
        fout.write("\n")
    fout.close()
    # save clip info
    start_t = format_t(float(sidx) / frames_cnt * duration)
    durat_t = format_t(float(eidx - sidx + 1) / frames_cnt * duration)
    fout = open(os.path.join(base_dir, "labels/{}.txt".format(vid_id)), 'a')
    fout.write("{} {} {} {} {}".format(clip_name, sidx, eidx, start_t, durat_t))
    # predict emotions labels
    for e in emotions:
        fout.write(" {} {:.5f} {}".format(e['index'], e['conf'], e['label']))
    fout.write("\n")
    fout.close()
    # cut video clips
    mkdir("{}/clips/{}".format(base_dir, vid_id))
    out_clip_path = os.path.join(base_dir, "clips/{}/{}.mp4".format(vid_id, clip_name))
    # -i after -ss fast but white screen
    # -i before -ss slow but clear
    os.system("ffmpeg -y -i {} -ss {} -t {} -c:v libx264 -c:a aac -strict \
              experimental -b:a 98k {}".format(vid_path, start_t, durat_t, out_clip_path))
    return 


# process a movie by detect faces
def process_a_movie(vid_path, base_dir, frames_cnt, duration, model, \
                    face_model, iou_thresh = 0.75, frame_thresh = (50, 300)):
    vid = imageio.get_reader(vid_path, 'ffmpeg')
    pre_face = None
    skip = 0
    continue_no_faces = 0
    sidx = eidx = 0
    sample_cnt = 0
    faces = []
    frames = []
    lands = []

    for idx, im in enumerate(vid):
        if idx % 200 == 0:
            print("idx = {}".format(idx))
        
        if skip > 0:
            skip -= 1
            continue
        img = Image.fromarray(im)
        cur_face, land = get_face(img, face_model)

        # no face detected, skip the next N frames
        if cur_face is None:
            skip = math.pow(2, continue_no_faces + 3)
            continue_no_faces = min(5, continue_no_faces + 1)
            # None face after serial faces
            if eidx - sidx + 1 > frame_thresh[0]:
                emotions = emotion_dectect(frames, faces, model)
                # valid clips with predict label
                if len(emotions) >= 1:
                    sample_cnt += 1
                    print("S0 {}: <{}-{}> ".format(sample_cnt, sidx, eidx, eidx - sidx + 1))
                    write_to_file(vid_path, sample_cnt, frames, faces, sidx, eidx, \
                                  emotions, base_dir, frames_cnt, duration, lands)
                    skip = max(skip, 16)
            sidx = eidx
            pre_face = None
            #print("skip:", skip)
            continue

        continue_no_faces = 0
        # first face
        if pre_face is None:
            pre_face = cur_face
            sidx = idx
            faces = []
            frames = []
            lands = []

        iou = cal_iou(pre_face, cur_face)
        pre_face = cur_face
        if iou >= iou_thresh:
            eidx = idx
            faces.append(cur_face)
            frames.append(img)
            lands.append(land)
            #print("idx = {} iou = {:.3f} len = {}".format(idx, iou, len(faces)))
            # out the max frame limit
            if eidx  - sidx + 1 >= frame_thresh[1]:
                emotions = emotion_dectect(frames, faces, model)
                # valid clips with predict label
                if len(emotions) >= 1:
                    sample_cnt += 1
                    print("S1 {}: <{}-{}> ".format(sample_cnt, sidx, eidx, eidx - sidx + 1))
                    write_to_file(vid_path, sample_cnt, frames, faces, sidx, eidx, \
                                  emotions, base_dir, frames_cnt, duration, lands)
                    skip = 16 
                pre_face = None
                sidx = eidx = idx       

        # cur face and pre face are not from same person
        # from cur face start, pre face = cur_face
        else:
            if eidx - sidx + 1 > frame_thresh[0]:
                emotions = emotion_dectect(frames, faces, model)
                # valid clips with predict label
                if len(emotions) >= 1:
                    sample_cnt += 1
                    print("S2 {}: <{}-{}> ".format(sample_cnt, sidx, eidx, eidx - sidx + 1))
                    write_to_file(vid_path, sample_cnt, frames, faces, sidx, eidx, emotions, base_dir, frames_cnt, duration, lands)
                    skip = 16
                    pre_face = None
                    sidx = eidx = idx
                    continue
            sidx = eidx = idx
            frames = [img,]
            faces = [cur_face,]
            lands = [land,]
    return


# get the label for manual annotation
def get_file_for_category_annotation(label_file, out_file, out_file2):
    if not os.path.exists(label_file):
        return
    fin = open(label_file)
    fout = open(out_file, 'w')
    fout2 = open(out_file2, 'a')
    maps = ['_', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    for line in fin.readlines():
        items = line.split()
        clip_id = items[0]
        frame_cnt = int(items[2]) - int(items[1]) + 1
        fout.write("{}".format(clip_id))
        fout2.write("{} \n".format(clip_id))
        
        emotions = items[5:]
        assert(len(emotions) % 3 == 0)
        
        scores = [0] * 7
        cnts = [0] * 7
        for i in range(0, len(emotions), 3):
            frame_index = int(emotions[i])
            cls = int(emotions[i+2])
            cnts[cls] += 1
            scores[cls] += float(emotions[i+1])

        d = {}
        for i,sc in enumerate(scores):
            if sc > 0:
                mean_sc = sc / cnts[i]
                d[maps[i]] = mean_sc
        
        pairs = sorted(d.items(), key=lambda x:x[1], reverse=True)
        for pair in pairs:
            fout.write(" {}({:.3f})".format(pair[0], pair[1]))
        fout.write("\n")
    fin.close()
    fout.close()
        

if __name__ == "__main__":
    movies = Movies()
    base_dir = movies.base_dir()
    mtcnn_model = init_model()
    fer_model = init_emotion_model()

    mkdir(base_dir + "/frames")
    mkdir(base_dir + "/faces")
    mkdir(base_dir + "/clips")
    mkdir(base_dir + "/labels")
    mkdir(base_dir + "/clean_labels")
    mkdir(base_dir + "/pick_labels")

    for movie_name in movies.names():
        movie_path = movies.name2path(movie_name)
        movie_id = movies.name2id(movie_name)

        duration, frame_cnt = get_length_of_movie(movie_name, movie_path, base_dir)
        print("Read Length:", movie_name, duration, frame_cnt)

        process_a_movie(movie_path, base_dir, frame_cnt, duration, fer_model, mtcnn_model)

        label_file = "{}/labels/{}.txt".format(base_dir, movie_id)
        out1 = "{}/clean_labels/{}.txt".format(base_dir, movie_id)
        out2 = "{}/pick_labels/{}.txt".format(base_dir, movie_id)
        get_file_for_category_annotation(label_file, out1, out2)
