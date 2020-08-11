# Process of Dataset Construction


## Prepare
* Movies
    * `base_dir = ${HOME}/FEID` for default, it can be changed in `config.py`
    * Prepare movie files in directory `${base_dir}/original`
    * Write the names of movies in the `line#15` of `config.py`.
* Python Package
    * python >= 3.5.2
    * imageio with ffmpeg, `pip3 install imageio`
    * PIL, `pip3 install pillow`
    * numpy, `pip3 install numpy`


## Clip Recommendation
```
    python clip_recommendation.py
```
Run the command and some files will be created:

* `${base_dir}/clips/${movie_id}/${movie_id}_${clip_id}.mp4`: save the extracted clips with continuous faces
* `${base_dir}/frames/${movie_id}/${movie_id}_${clip_id}/${frame_id}.jpg`: save the frames of clips
* `${base_dir}/faces/${movie_id}_${clip_id}.txt` save the face rectangles of clips
* `${base_dir}/labels/${movie_id}.txt` save the extract information (duration, frame count, fer results)
* `${base_dir}/clean_labels/${movie_id}.txt` save the statistic results of fer prediction, for refer of category annotation
* `${base_dir}/pick_labels/${movie_id}.txt` save the init expression category annotation

## Category Annotation



## Key Frame Selection


## Intensity Annotation
