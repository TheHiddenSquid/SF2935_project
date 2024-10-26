import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Average of 78 % correct (unseen data)
# 78 % if k=3 and k=5


# 78.2 %    no weight
# 79.1 %    custom weight   [1.64, 1.74, 0.67, 0.98, 0.16, 1.69, 1.49, 0.67, 1.18, 0.96, 0,87]
# 81.1 %    custom weight   [0.82, 1.02, 0.26, 1.35, 0.15, 2, 0.98, 0.84, 1.13, 0.88, 0]

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


def classify(song: Song, training_data: List[Song], k:int = 3) -> bool:
    attributes = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence","tempo"]
    
    bounds_dict = {"danceability":[0,1], "energy":[0,1], "key":[0,11],"loudness":[-60,0], "mode":[0,1], "speechiness":[0,1], "acousticness":[0,1], "instrumentalness":[0,1], "liveness":[0,1], "valence":[0,1], "tempo":[20,200]}

    ans_tuples = []

    for other in training_data:
        distance = []
        for attr in attributes:
            distance.append( abs(getattr(other, attr) - getattr(song, attr)) / (bounds_dict[attr][1]-bounds_dict[attr][0]) )

        #distance = sum(distance)                           # taxicab distance
        #distance = sum(x**2 for x in distance) ** (1/2)    # Euclidean distance
        #distance = max(distance)                            # Chebyshev distance

        opt_weights =  [0.82, 1.02, 0.26, 1.35, 0.15, 2, 0.98, 0.84, 1.13, 0.88, 0]

        distance = sum(distance[i]*opt_weights[i] for i in range(11))


        ans_tuples.append((other, distance))

    ans_tuples.sort(key = lambda x: x[1])

    liked = 0
    disliked = 0 
    for i in range(k):
        if ans_tuples[i][0].Label == 1:
            liked += 1
        else:
            disliked += 1

    return 1 if liked > disliked else 0


def random_training_data_test(songs: List[Song], no_tests:int) -> float:
    random.seed(1234)
    score = 0
    training_data = []
    testing_data = []
    
    for i in range(no_tests):
        #print(i)
        
        testing_data = random.sample(songs, 100)
        training_data = [x for x in songs if x not in testing_data]

        correct = 0
        tot = 0

        for song in testing_data:
            ans = classify(song, training_data, 3)
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
    
    
    ans = random_training_data_test(songs, 10)
    print("regular:", ans)
   
    
    

        

if __name__ == "__main__":
    main()