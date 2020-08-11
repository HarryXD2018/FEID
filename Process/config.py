import os


class Movies(object):

    def __init__(self):
        self._base_dir = os.environ['HOME'] + "/FEID/"
        mkdir(self._base_dir)

        # candidate movies in dir $HOME/FEID/original
        self._names = [
            "movie-1.mp4",
            "movie-2.mp4",
            "movie-3.mp4",
            "movie-4.mp4",
            "movie-5.mp4",
        ]
        for name in self._names:
            assert(os.path.exists(self._base_dir + "original/" + name))
        self._emotions = ["_", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
         
    def base_dir(self):
        return self._base_dir

    def names(self):
        return self._names

    def name2id(self, name):
        return name.split('.')[0]

    def name2path(self, name):
        return self._base_dir + "original/" + name

    def emotions(self):
        return self._emotions 


def mkdir(dd):
    if not os.path.exists(dd):
        os.makedirs(dd)


def sort_listdir(dd):
    ds = os.listdir(dd)
    ds.sort()
    return ds
