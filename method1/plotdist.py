import matplotlib.pyplot as plt
from typing import List
from scipy import stats
import numpy as np

class Song():
    def __init__(self, danceability, energy, key, loudness, mode, speechiness,
                 acousticness, instrumentalness, liveness, valence, tempo, Label):
        self.danceability = danceability
        self.energy = energy
        self.key = key
        self.loudness = loudness
        self.mode = mode
        self.speechiness = speechiness
        self.acousticness = acousticness
        self.instrumentalness = instrumentalness
        self.liveness = liveness
        self.valence = valence
        self.tempo = tempo
        self.Label = Label


def get_songs(filename: str) -> List[Song]:
    songs = []
    with open(filename) as f:
        f.readline() # skip first
        for line in f:
            songdata = [eval(x) for x in line.strip().split(",")]
            songs.append(Song(*songdata))

    del songs[84]       # outlier
    del songs[93]       # outlier
    return songs


def main():
    songs = get_songs("project_train.csv")
    liked_songs = [x for x in songs if x.Label == 1]
    disliked_songs = [x for x in songs if x.Label == 0]


    # NEEDS CUSTOM: key, mode
    bins = 50
    choices = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence","tempo"]

    choices = ["speechiness","liveness","instrumentalness"]
    

    bounds_dict = {"danceability":[0,1], "energy":[0,1],"loudness":[-60,0], "speechiness":[0,1], "acousticness":[0,1], "instrumentalness":[0,1], "liveness":[0,1], "valence":[0,1], "tempo":[20,200]}

    for choice in choices:
        
        m = bounds_dict[choice][0]
        M = bounds_dict[choice][1]

        liked_data = [(getattr(x, choice)-m)/(M-m) for x in liked_songs]
        disliked_data = [(getattr(x, choice)-m)/(M-m) for x in disliked_songs]
    


        plt.clf()
        plt.subplot(1,2,1)
        plt.hist(liked_data, bins=bins, alpha=0.5, label="liked", density=True, color="C0")
        # a,b, *args = stats.fit(stats.beta, liked_data, [(0, 100), (0, 100)]).params
        # rv = stats.beta(a, b)


        rv = stats.gaussian_kde(liked_data)

        x = np.linspace(0,1,100)
        plt.plot(x, rv.pdf(x), 'b-', lw=2, label='estimate')
        plt.legend(loc="upper right")


        plt.subplot(1,2,2)
        plt.hist(disliked_data, bins=bins, alpha=0.5, label="disliked", density=True, color="C1")
        # a,b, *args = stats.fit(stats.beta, disliked_data, [(0, 100), (0, 100)]).params
        # rv = stats.beta(a, b)
        

        rv = stats.gaussian_kde(disliked_data)

        x = np.linspace(0,1,100)
        plt.plot(x, rv.pdf(x), 'r-', lw=2, label='estimate')
        plt.legend(loc="upper right")
        
        
        plt.suptitle(choice)
        plt.show()
        #plt.savefig(f"{choice}2.png")
    

    


if __name__ == "__main__":
    main()