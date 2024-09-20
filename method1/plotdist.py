import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

class Song():
    def __init__(self, data):
        self.danceability = eval(data[0])
        self.energy = eval(data[1])
        self.key = eval(data[2])
        self.loudness = eval(data[3])
        self.mode = eval(data[4])
        self.speechiness = eval(data[5])
        self.acousticness = eval(data[6])
        self.instrumentalness = eval(data[7])
        self.liveness = eval(data[8])
        self.valence = eval(data[9])
        self.tempo = eval(data[10])
        self.Label = eval(data[11])


def get_songs():
    songs = []
    with open("project_train.csv") as f:
        skip = True
        for line in f:
            if skip:
                skip = False
                continue
            songs.append(Song(line.strip().split(",")))

    del songs[84]       # outlier
    del songs[93]       # outlier
    return songs


def main():
    songs = get_songs()
    liked_songs = [x for x in songs if x.Label == 1]
    disliked_songs = [x for x in songs if x.Label == 0]


    # NEEDS CUSTOM: key, mode
    bins = 20
    choices = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence","tempo"]
    

    bounds_dict = {"danceability":[0,1], "energy":[0,1],"loudness":[-60,0], "speechiness":[0,1], "acousticness":[0,1], "instrumentalness":[0,1], "liveness":[0,1], "valence":[0,1], "tempo":[20,200]}

    for choice in choices:
        
        m = bounds_dict[choice][0]
        M = bounds_dict[choice][1]

        liked_data = [(getattr(x, choice)-m)/(M-m) for x in liked_songs]
        disliked_data = [(getattr(x, choice)-m)/(M-m) for x in disliked_songs]
    


        plt.clf()
        plt.subplot(1,2,1)
        plt.hist(liked_data, bins=bins, alpha=0.5, label="liked", density=True, color="C0")
        a,b, *args = stats.fit(stats.beta, liked_data, [(0, 100), (0, 100)]).params
        x = np.linspace(0,1,100)
        rv = stats.beta(a, b)
        rv = stats.gaussian_kde(liked_data)
        plt.plot(x, rv.pdf(x), 'b-', lw=2, label='estimate')
        plt.legend(loc="upper right")


        plt.subplot(1,2,2)
        plt.hist(disliked_data, bins=bins, alpha=0.5, label="disliked", density=True, color="C1")
        a,b, *args = stats.fit(stats.beta, disliked_data, [(0, 100), (0, 100)]).params
        x = np.linspace(0,1,100)
        rv = stats.beta(a, b)
        rv = stats.gaussian_kde(disliked_data)
        plt.plot(x, rv.pdf(x), 'r-', lw=2, label='estimate')
        plt.legend(loc="upper right")
        
        
        plt.suptitle(choice)
        plt.show()
    

    


if __name__ == "__main__":
    main()