import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Best
# 79.2% sceme = 1, k = 32
# 79.4% sceme = 2, k = 5

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


def classify(song: Song, training_data: List[Song],k=5,  weight_sceme:int = 1) -> bool:
    attributes = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence","tempo"]
    
    bounds_dict = {"danceability":[0,1], "energy":[0,1], "key":[0,11],"loudness":[-60,0], "mode":[0,1], "speechiness":[0,1], "acousticness":[0,1], "instrumentalness":[0,1], "liveness":[0,1], "valence":[0,1], "tempo":[20,200]}

    ans_tuples = []

    for other in training_data:
        distance = []
        for attr in attributes:
            distance.append( abs(getattr(other, attr) - getattr(song, attr)) / (bounds_dict[attr][1]-bounds_dict[attr][0]) )

        distance = sum(distance)                           # Taxicab distance, no weights

        ans_tuples.append((other, distance))

    ans_tuples.sort(key = lambda x: x[1])
    
    # Special case if we have known song
    if ans_tuples[0][1] == 0:
        return ans_tuples[0][0].Label

    if weight_sceme == 1:
        liked = 0
        disliked = 0 
        for i in range(k):
            if ans_tuples[i][0].Label == 1:
                liked += 1 / (i+1)
            else:
                disliked += 1 / (i+1)

    if weight_sceme == 2:
        liked = 0
        disliked = 0 
        for i in range(k):
            if ans_tuples[i][0].Label == 1:
                liked += 1 / ans_tuples[i][1]
            else:
                disliked += 1 / ans_tuples[i][1]

    return 1 if liked > disliked else 0


def random_training_data_test(songs: List[Song], k:int, weight_sceme:int, no_tests:int) -> float:
    random.seed(1234)
    score = 0
    
    for _ in range(no_tests):        
        testing_data = random.sample(songs, 100)
        training_data = [x for x in songs if x not in testing_data]

        correct = 0
        tot = 0

        for song in testing_data:
            ans = classify(song, training_data, k, weight_sceme)
            if ans == song.Label:
                correct += 1
            tot += 1

        score += correct/tot

    return 100 * score/no_tests


def main():

    songs = get_songs("../project_train.csv")

    #best
    print(random_training_data_test(songs, 32, 1, 100))
    print(random_training_data_test(songs, 5, 2, 100))
    

    # Graph for varying k
    ks = []
    ans1 = []
    ans2 = []
    for k in range(1,100):
        ks.append(k)
        ans1.append(random_training_data_test(songs, k, 1, 100))
        ans2.append(random_training_data_test(songs, k, 2, 100))

    plt.plot(ks, ans1, label="version 1")
    plt.plot(ks, ans2, label="version 2")
    plt.xlabel("k")
    plt.ylabel("Correctness (%)")
    plt.title("Correctness vs k")
    plt.legend(loc="upper right")
    plt.show()

    

if __name__ == "__main__":
    main()