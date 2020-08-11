
# Process of Dataset Construction


## Prepare
* Movies
    * `base_dir = ${HOME}/FEID` for default, it can be changed in `config.py`
    * Prepare movie files in directory `${base_dir}/original`
    * Write the names of movies in the `line#12` of `config.py`.
* Package
    * python >= 3.5.2
    * imageio with ffmpeg, `pip3 install imageio`
    * PIL, `pip3 install pillow`
    * numpy, `pip3 install numpy`
    * dlib, `pips install dlib`
    * opencv, `pip3 install opencv-python`
* Model
    * Download [landmarks model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), and put it in current directory


## Clip Recommendation
```
    python clip_recommendation.py
```
Run the command and some files will be created:

* `${base_dir}/clips/${movie_id}/${clip_id}.mp4`: saves the extracted clips with continuous faces
* `${base_dir}/frames/${movie_id}/${clip_id}/${frame_id}.jpg` saves the frames of clips
* `${base_dir}/faces/${clip_id}.txt` saves the face rectangles of clips
* `${base_dir}/labels/${movie_id}.txt` saves the extract information (duration, frame count, FER results)
* `${base_dir}/clean_labels/${movie_id}.txt` saves the statistic results of FER prediction, for refer of category annotation
* `${base_dir}/pick_labels/${movie_id}.txt` saves the init expression category annotation

## Category Annotation
According to the actual expression and the prediction of the FER model, the annotate expression category for all extracted clips.

* Annotation file: `${base_dir}/pick_labels/${movie_id}.txt`, each line represents a clip
* Annotation format: If a clip reveals basic expression, change the corresponding line with `${clip_id} ${expresion_category} ${actor_name} ${actor_gender} ${actor_age}`, otherwise no operation
    * `expression_category`: 1-Anger, 2-Disgust, 3-Fear, 4-Happiness, 5-Sadness, 6-Surprise
    * `actor_name`: if not access, annotate with `Unknown`
    * `actor_age`: if not access, annotate with 0

```
    python category_annotation
```
After manual annotation, run the command and some files will be created:

* `${base_dir}/dataset` is for the constructed dataset, it includes 6 subdirs related to 6 basic expressions
* `${base_dir}/landmarks/${clip_id}.txt` saves the landmarks detected by dlib
* `${base_dir}/dataset/${emotion}/${clip_id}` is the base directory of a clip sample, it includes:
    * `${clip_id}.mp4`: clip file
    * `annotation.xml`: annotation xml file
    * `frames/`: directory for saving frames
    * `normalize_256/`: normalized faces

## Key Frame Selection
Train an effective FER model on current dataset, and put the prediction scores in `${base_dir}/predcit_score/`
```
    python key_frame_selection.py
```
Run the command and some files will be created:

* `${base_dir}/predict_score_smooth/`: smooth prediction scores
* `${base_dir}/key_frames/`: indexs of key frames

## Intensity Annotation
This step is a manual step. We display the clips in a website for intensity annotation. After all annotations are complete, put the label dir in `${base_dir}/label/`
```
    python intensity annotation
```
Run the command and generate annotated intensities in `${base_dir}/intensity/`
