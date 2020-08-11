from config import *
from xml.dom.minidom import parse
from xml.dom import minidom
import cv2
import dlib
import numpy as np


# parse the label file
def get_label_d(label_file):
    d = {}
    with open(label_file) as f:
        for line in f.readlines():
            items = line.split()
            d[items[0]] = {
                "start_frame_index": int(items[1]),
                "end_frame_index": int(items[2]),
                "frame_cnt": int(items[2]) - int(items[1]) + 1,
                "start_time": items[3],
                "duration_time": items[4],
            }
    return d


# parse the annotation file
def get_manual_d(manual_file):
    d = {}
    with open(manual_file) as f:
        for line in f.readlines():
            items = line.split()
            if len(items) > 1:
                assert(len(items) == 5)
                assert(items[3] in ["Male", "Female"])
                d[items[0]] = {
                    "emotion_label": int(items[1]),
                    "name_of_actor": items[2].replace("_", " "),
                    "gender_of_actor": items[3],
                    "age_of_actor": int(items[4])
                }
    return d


#  parse the face file
def get_face_l(face_file):
    l = []
    with open(face_file) as f:
        for line in f.readlines():
            items = line.split()
            l.append({
                "frame": items[0],
                "x1": float(items[1]),
                "y1": float(items[2]),
                "x2": float(items[3]),
                "y2": float(items[4]),
            })
    return l


def gen_meta_tag(doc, name, value):
    meta_tag = doc.createElement("Metatag")
    meta_tag.setAttribute("name", name)
    meta_tag.setAttribute("value", str(value))
    return meta_tag


def simple_node(doc, tag, value):
    n = doc.createElement(tag)
    n.appendChild(doc.createTextNode(str(value)))
    return n


# create the face rectangle node
def get_face_node(doc, faces):
    N = doc.createElement("Faces")
    for face in faces:
        n = doc.createElement("Face")
        n.setAttribute("frame", face['frame'])
        n.appendChild(simple_node(doc, "x1", face["x1"]))
        n.appendChild(simple_node(doc, "y1", face["y1"]))
        n.appendChild(simple_node(doc, "x2", face["x2"]))
        n.appendChild(simple_node(doc, "y2", face["y2"]))
        N.appendChild(n)
    return N


# according to the annotation, generate the xml and tidy the dataset
def gen_xml(movie_id, emotions, base_dir, dataset_dir):
    label_file = "{}/labels/{}.txt".format(base_dir, movie_id)
    manual_file = "{}/pick_labels/{}.txt".format(base_dir, movie_id)
    frame_dir = "{}/frames/{}".format(base_dir, movie_id)
    clip_dir = "{}/clips/{}".format(base_dir, movie_id)
    face_dir = "{}/faces".format(base_dir)

    label_d = get_label_d(label_file)
    manual_d = get_manual_d(manual_file)

    for key in manual_d.keys():
        clip_id = key
        vid_name = movie_name.replace("_", " ")
        emotion_label = manual_d[key]["emotion_label"]
        actor_name = manual_d[key]["name_of_actor"]
        actor_gender = manual_d[key]["gender_of_actor"]
        actor_age = manual_d[key]["age_of_actor"]
        start_frame_index = label_d[key]["start_frame_index"]
        end_frame_index = label_d[key]["end_frame_index"]
        frame_cnt = label_d[key]["frame_cnt"]
        start_time = label_d[key]["start_time"]
        duration_time = label_d[key]["duration_time"]

        # generate xml label file
        doc = minidom.Document()
        N_anno = doc.createElement("Annotation")
        N_anno.appendChild(gen_meta_tag(doc, "MovieTitle", vid_name))
        N_anno.appendChild(gen_meta_tag(doc, "ClipName", clip_id + ".mp4"))
        N_anno.appendChild(gen_meta_tag(doc, "Emotion", emotion_label))
        N_anno.appendChild(gen_meta_tag(doc, "NameOfActor", actor_name))
        N_anno.appendChild(gen_meta_tag(doc, "GenderOfActor", actor_gender))
        N_anno.appendChild(gen_meta_tag(doc, "AgeOfActor", actor_age))
        N_anno.appendChild(gen_meta_tag(doc, "FrameStartIndex", start_frame_index))
        N_anno.appendChild(gen_meta_tag(doc, "FrameEndIndex", end_frame_index))
        N_anno.appendChild(gen_meta_tag(doc, "StartTime", start_time))
        N_anno.appendChild(gen_meta_tag(doc, "DurationTime", duration_time))

        # face annotation
        face_src = "{}/{}.txt".format(face_dir, clip_id)
        faces = get_face_l(face_src)
        N_anno.appendChild(get_face_node(doc, faces))
        doc.appendChild(N_anno)

        des_base_dir = "{}/{}/{}".format(dataset_dir, emotions[emotion_label], clip_id)
        mkdir(des_base_dir)
        xml_des = "{}/annotation.xml".format(des_base_dir)
        with open(xml_des, 'w') as fh:
            doc.writexml(fh, indent='', addindent='\t', newl='\n')

        # copy frames and clip
        frame_src = "{}/{}".format(frame_dir, clip_id)
        clip_src = "{}/{}.mp4".format(clip_dir, clip_id)

        frame_des = "{}/frames".format(des_base_dir)
        if os.path.exists(frame_des):
            os.system("rm -r {}".format(frame_des))
        clip_des = "{}/{}.mp4".format(des_base_dir, clip_id)
        os.system("cp {} {}".format(clip_src, clip_des))
        os.system("cp -r {} {}".format(frame_src, frame_des))

        print(vid_name, clip_id, emotion_label, actor_name, actor_gender, actor_age, start_frame_index, end_frame_index, frame_cnt, start_time, duration_time)


# parse label xml file
def parse_xml_face(fname):
    with open(fname, 'r') as fh:
        doc = minidom.parse(fh)
        root = doc.documentElement
        faces = root.getElementsByTagName('Faces')[0].getElementsByTagName('Face')
        d = {}
        for face in faces:
            x1 = float(face.getElementsByTagName('x1')[0].childNodes[0].data)
            y1 = float(face.getElementsByTagName('y1')[0].childNodes[0].data)
            x2 = float(face.getElementsByTagName('x2')[0].childNodes[0].data)
            y2 = float(face.getElementsByTagName('y2')[0].childNodes[0].data)
            d[face.getAttribute('frame')] = [x1, y1, x2, y2]
        return d


def dlib_land(img, f, Predictor):
    rect = dlib.rectangle(int(f[0]), int(f[1]), int(f[2]), int(f[3]))
    results = Predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in results.parts()])
    return landmarks


def get_landmarks(clip_dir, land_file, Predictor):
    faces = parse_xml_face(os.path.join(clip_dir, "annotation.xml"))
    frame_ids = sort_listdir(os.path.join(clip_dir, "frames"))

    fout = open(land_file, 'w')
    for frame_id in frame_ids:
        img = cv2.imread("{}/frames/{}".format(clip_dir, frame_id))
        f = faces[frame_id]
        landmarks = dlib_land(img, f, Predictor)
        fout.write("{} {} {} {} {}".format(frame_id, f[0], f[1], f[2], f[3]))
        for pts in landmarks:
            fout.write(" {} {}".format(pts[0], pts[1]))
        fout.write("\n")
    fout.close()


def get_same_triangle(A, B):
    A = A[:, np.newaxis]
    B = B[:, np.newaxis]
    AB = B - A
    theta = 60 * np.pi / 180
    T = np.float32([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    C = A + np.matmul(T, AB)
    return C[:, 0]


def key_pt(land):
    p1 = land[39]
    p2 = land[42]
    p3 = get_same_triangle(p1, p2)
    return np.array([p1, p2, p3])


# normalize faces according to key points
def norm_faces(sample_base_dir, land_file):
    target_dir = os.path.join(sample_base_dir, "normalize_256")
    mkdir(target_dir)
    
    d = 256
    lines = open(land_file).readlines()
    all_landmarks = []
    face_rects = []
    img_paths = []
    for line in lines:
        items = line.split()
        assert(len(items) == 141)
        img_paths.append("{}/frames/{}".format(sample_base_dir, items[0]))
        face_rects.append(np.float32([float(x) for x in items[1:5]]))
        all_landmarks.append(np.float32([float(x) for x in items[5:]]).reshape(68, 2))

    # align with mean face and mean landmarks
    w1, h1, w2, h2 = np.mean(np.array(face_rects), axis=0)
    mean_land = np.mean(np.array(all_landmarks), axis=0)
    des = key_pt(mean_land)
    for pt in des:
        pt[0] = (pt[0] - w1) / (w2 - w1) * d
        pt[1] = (pt[1] - h1) / (h2 - h1) * d
    
    # for each frame
    for idx, landmarks in enumerate(all_landmarks):
        img = cv2.imread(img_paths[idx])
        M = cv2.getAffineTransform(key_pt(landmarks), des)
        warped = cv2.warpAffine(img, M, (d, d), borderValue=0.0)
        cv2.imwrite(os.path.join(target_dir, img_paths[idx].split('/')[-1]), warped)


if __name__ == "__main__":
    movies = Movies()
    emotions = movies.emotions()

    base_dir = movies.base_dir()
    dataset_dir = base_dir + "dataset/"
    land_dir = base_dir + "landmarks/"
    mkdir(land_dir)

    Predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    for i in range(1, len(emotions)):
        mkdir(dataset_dir + emotions[i])

    # tidy the clips with expression
    for movie_name in movies.names():
        movie_path = movies.name2path(movie_name)
        movie_id = movies.name2id(movie_name)
        gen_xml(movie_id, emotions, base_dir, dataset_dir)
     
    for emotion in emotions[1:]:
        emotion_dir = os.path.join(dataset_dir, emotion)
        clip_dirs = sort_listdir(emotion_dir)

        for clip_dir in clip_dirs:
            clip_id = clip_dir.split('/')[-1]
            clip_path = os.path.join(emotion_dir, clip_dir)
            land_file = os.path.join(land_dir, clip_id + ".txt")

            # get the 68 landmarks
            get_landmarks(clip_path, land_file, Predictor)

            # normalize faces
            norm_faces(clip_path, land_file)
