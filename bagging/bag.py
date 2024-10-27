import numpy as np
from typing import List
import random
from scipy import stats
import matplotlib.pyplot as plt

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


def get_songs(filename: str) -> List[List[float]]:
    songs = []
    with open(filename) as f:
        f.readline() # skip first
        for line in f:
            songdata = [eval(x) for x in line.strip().split(",")]
            songs.append(songdata)

    del songs[84]       # outlier
    del songs[93]       # outlier
    return songs


# W-nn functions
def classify_wnn(song: Song, training_data: List[Song]) -> bool:
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


    liked = 0
    disliked = 0 
    for i in range(32):
        if ans_tuples[i][0].Label == 1:
            liked += 1 / (i+1)
        else:
            disliked += 1 / (i+1)


    return 1 if liked > disliked else 0


# Naive Bayes functions
def generate_pmfs(songs, bins):
    liked_songs = [x for x in songs if x.Label == 1]
    disliked_songs = [x for x in songs if x.Label == 0]
    
    liked_pmfs = {}
    disliked_pmfs = {}

    # Discretisation of continuous rvs:
    attributes = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence","tempo"]
    bounds_dict = {"danceability":[0,1], "energy":[0,1],"loudness":[-60,0], "speechiness":[0,1], "acousticness":[0,1], "instrumentalness":[0,1], "liveness":[0,1], "valence":[0,1], "tempo":[20,200]}

    for attr in attributes:
        liked_bin_count = [0]*bins
        disliked_bin_count = [0]*bins

        m = bounds_dict[attr][0]    # For rescaling to [0,1]
        M = bounds_dict[attr][1]    # For rescaling to [0,1]
        liked_data = [getattr(x, attr) for x in liked_songs]
        disliked_data = [getattr(x, attr) for x in disliked_songs]

        for dataset, counter in [(liked_data, liked_bin_count), (disliked_data, disliked_bin_count)]:
            for bin in range(bins):
                left = bin * (M-m)/bins + m
                right = (bin+1) * (M-m)/bins + m

                for point in dataset:
                    if point >= M:
                       counter[-1] += 1 
                    elif point >= left and point < right:
                        counter[bin] += 1
        
        liked_pmfs[attr] = [x/len(liked_data) for x in liked_bin_count]

        disliked_pmfs[attr] = [x/len(disliked_data) for x in disliked_bin_count]


    # Custum for "mode":
    attr = "mode"
    liked_data = [getattr(x, attr) for x in liked_songs]
    disliked_data = [getattr(x, attr) for x in disliked_songs]

    liked_pmfs[attr] = [liked_data.count(x)/len(liked_data) for x in [0,1]]
    disliked_pmfs[attr] = [disliked_data.count(x)/len(disliked_data) for x in [0,1]]


    # Custom for "key"
    attr = "key"
    liked_data = [getattr(x, attr) for x in liked_songs]
    disliked_data = [getattr(x, attr) for x in disliked_songs]

    liked_pmfs[attr] = [liked_data.count(x)/len(liked_data) for x in range(12)]
    disliked_pmfs[attr] = [disliked_data.count(x)/len(disliked_data) for x in range(12)]
    

    return liked_pmfs, disliked_pmfs

def eval_pdf(song, pmfs, bins):
    tot = 1

    # Count all discretizised distributions
    attributes = ["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence","tempo"]
    bounds_dict = {"danceability":[0,1], "energy":[0,1],"loudness":[-60,0], "speechiness":[0,1], "acousticness":[0,1], "liveness":[0,1], "valence":[0,1], "tempo":[20,200]}

    for attr in attributes:
        m = bounds_dict[attr][0]
        M = bounds_dict[attr][1]
        point = getattr(song, attr)

        bin = None
        for i in range(bins):
            left = i * (M-m)/bins + m
            right = (i+1) * (M-m)/bins + m
            if point >= M:
               bin = -1
            elif point >= left and point < right:
                bin = i
        
        tot *= pmfs[attr][bin]

    # Count the discrete distributions
    attr = "mode"
    tot *= pmfs[attr][getattr(song, attr)]

    attr = "key"
    tot *= pmfs[attr][getattr(song, attr)]

    return tot

def classify_naive_bayes(song, liked_pdfs, disliked_pdfs,  bins, p_liked=0.5, p_disliked=0.5):
    if eval_pdf(song, liked_pdfs, bins) * p_liked > eval_pdf(song, disliked_pdfs, bins) * p_disliked:
        ans = 1
    else:
        ans = 0
    return ans


# LDA functions
def generate_LDA_rvs(training_data):
    songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in training_data], axis=1)
    cov = np.cov(songs_array)

    liked_songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in training_data if song[-1] == 1], axis=1)
    liked_mean = np.mean(liked_songs_array, axis=1)
    liked_rv = stats.multivariate_normal(liked_mean, cov)

    disliked_songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in training_data if song[-1] == 0], axis=1)
    disliked_mean = np.mean(disliked_songs_array, axis=1)
    disliked_rv = stats.multivariate_normal(disliked_mean, cov)

    return liked_rv, disliked_rv

def classify_LDA(song, liked_rv, disliked_rv) -> bool:
    if liked_rv.pdf(song) > disliked_rv.pdf(song):
        return 1
    else:
        return 0


# tests
def plot_ns_test():
    random.seed("random")

    songs = get_songs("../project_train.csv")
    n = 300

    ns = []
    ys = []
    for n in range(100, 400):

        ns.append(n)

        scores = []
        for _ in range(100):
            training_data = random.sample(songs, 400)
            testing_data = [song for song in songs if song not in training_data]

            # Generate 3 random datasets

            bins = int(np.floor(2.5 + 0.0233333 * n))

            model1_training = []
            model2_training = []
            model3_training = []
            for _ in range(n):
                model1_training.append(random.choice(training_data))
                model2_training.append(Song(*random.choice(training_data)))
                model3_training.append(Song(*random.choice(training_data)))

            
            # Train the models
            liked_rv, disliked_rv = generate_LDA_rvs(model1_training)
            liked_pmfs, disliked_pmfs = generate_pmfs(model2_training, bins)
        

            # Classify using the models
            correct = 0
            for song in testing_data:
                ans1 = classify_LDA(song[:11], liked_rv, disliked_rv)
                ans2 = classify_naive_bayes(Song(*song), liked_pmfs, disliked_pmfs,  bins)
                ans3 = classify_wnn(Song(*song), model3_training)

                #print(ans1, ans2, ans3, song[-1])

                if ans1+ans2+ans3 >=2:
                    ans = 1
                else:
                    ans = 0
                
                if ans == song[-1]:
                    correct += 1

            scores.append(correct/len(testing_data))

        ys.append(np.mean(scores))

    plt.plot(ns, ys)
    plt.xlabel("n")
    plt.ylabel("Correctness (%)")
    plt.title("Correctness vs n")
    plt.show()


def main():
    
    random.seed("random")

    songs = get_songs("../project_train.csv")
    n = 360


    scores = []
    for _ in range(1):
        training_data = random.sample(songs, 400)
        testing_data = [song for song in songs if song not in training_data]

        # Generate 3 random datasets
        model1_training = []
        model2_training = []
        model3_training = []
        for _ in range(n):
            model1_training.append(random.choice(training_data))
            model2_training.append(Song(*random.choice(training_data)))
            model3_training.append(Song(*random.choice(training_data)))

        
        # Train the models
        bins = int(np.floor(2.5 + 0.0233333 * n))
        liked_rv, disliked_rv = generate_LDA_rvs(model1_training)
        liked_pmfs, disliked_pmfs = generate_pmfs(model2_training, bins)
    

        # Classify using the models
        correct = 0
        for song in testing_data:
            ans1 = classify_LDA(song[:11], liked_rv, disliked_rv)
            ans2 = classify_naive_bayes(Song(*song), liked_pmfs, disliked_pmfs,  bins)
            ans3 = classify_wnn(Song(*song), model3_training)

            # Nice comparison to look at
            #print(ans1, ans2, ans3, song[-1])

            if ans1+ans2+ans3 >=2:
                ans = 1
            else:
                ans = 0
            
            if ans == song[-1]:
                correct += 1

        scores.append(correct/len(testing_data))

    print(np.mean(scores))


if __name__ == "__main__":
    main()