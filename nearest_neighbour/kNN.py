import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 79.04 %    no weight
# 81.23 %    custom weight   [0.82, 1.02, 0.26, 1.35, 0.15, 2, 0.98, 0.84, 1.13, 0.88, 0]

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


def classify(song: Song, training_data: List[Song], k:int = 5, metric:int = 1) -> bool:
    attributes = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence","tempo"]
    
    bounds_dict = {"danceability":[0,1], "energy":[0,1], "key":[0,11],"loudness":[-60,0], "mode":[0,1], "speechiness":[0,1], "acousticness":[0,1], "instrumentalness":[0,1], "liveness":[0,1], "valence":[0,1], "tempo":[20,200]}

    ans_tuples = []

    for other in training_data:
        distances = []
        for attr in attributes:
            distances.append( abs(getattr(other, attr) - getattr(song, attr)) / (bounds_dict[attr][1]-bounds_dict[attr][0]) )

        #weights =  [0.82, 1.02, 0.26, 1.35, 0.15, 2, 0.98, 0.84, 1.13, 0.88, 0]     # Weights
        weights =  [1]*11                                                          # No weights

        scaled_distances = [distances[i]*weights[i] for i in range(11)] 

        if metric == 1:
            distance = sum(scaled_distances)                           # Taxicab distance
        if metric == 2:
            distance = sum(x**2 for x in scaled_distances) ** (1/2)    # Euclidean distance
        if metric == 3:
            distance = max(scaled_distances)                            # Chebyshev distance

        ans_tuples.append((other, distance))

    ans_tuples.sort(key = lambda x: x[1])

    # Special case if we have known song
    if ans_tuples[0][1] == 0:
        return ans_tuples[0][0].Label

    liked = 0
    disliked = 0 
    for i in range(k):
        if ans_tuples[i][0].Label == 1:
            liked += 1 
        else:
            disliked += 1 

    return 1 if liked > disliked else 0


def random_training_data_test(songs: List[Song], k:int, metric:int, no_tests:int) -> float:
    random.seed(1234)
    score = 0
    
    for _ in range(no_tests):        
        testing_data = random.sample(songs, 100)
        training_data = [x for x in songs if x not in testing_data]

        correct = 0
        tot = 0

        for song in testing_data:
            ans = classify(song, training_data, k, metric)
            if ans == song.Label:
                correct += 1
            tot += 1

        score += correct/tot

    return 100 * score/no_tests


def optimize_one_weight(songs: List[Song], weights, index):
    xs = np.linspace(0,2.5,50)
    ys = []

    for x in xs:
        weights[index] = x
    
        ans = random_training_data_test(songs, 20, weights)
        ys.append(ans)
    
    plt.plot(xs,ys, label=fr"$X_{{{index+1}}}$")
    plt.xlabel("weight")
    plt.ylabel("correctness %")
    #plt.show()


def main():

    songs = get_songs("../project_train.csv")
    print(random_training_data_test(songs, 5, 1, 1000))


if __name__ == "__main__":
    main()
