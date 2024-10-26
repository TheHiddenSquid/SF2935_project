import random
import numpy as np
from typing import List
import scipy.stats as stats

#QDA: 79,16 %
#LDA: 79,96 %

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


def classify(song, liked_rv, disliked_rv) -> bool:
    if liked_rv.pdf(song) > disliked_rv.pdf(song):
        return True
    else:
        return False


def generate_QDA_rvs(training_data):
    liked_songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in training_data if song[-1] == 1], axis=1)
    liked_mean = np.mean(liked_songs_array, axis=1)
    liked_cov = np.cov(liked_songs_array)
    liked_rv = stats.multivariate_normal(liked_mean, liked_cov)

    disliked_songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in training_data if song[-1] == 0], axis=1)
    disliked_mean = np.mean(disliked_songs_array, axis=1)
    disliked_cov = np.cov(disliked_songs_array)
    disliked_rv = stats.multivariate_normal(disliked_mean, disliked_cov)

    return liked_rv, disliked_rv


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


def test_methods(songs, no_tests, method):
    results = []
    for _ in range(no_tests):
        training = random.sample(songs, 400)
        testing = [song for song in songs if song not in training]

        if method == "QDA":
            liked_rv, disliked_rv = generate_QDA_rvs(training)
        elif method == "LDA":
            liked_rv, disliked_rv = generate_LDA_rvs(training)
        else:
            raise ValueError("Method not supported")
        
        correct = 0
        for song in testing:
            ans = classify(song[:11], liked_rv, disliked_rv)
            if ans == song[-1]:
                correct += 1
        
        results.append(correct)

    return np.mean(results)


def main():

    songs = get_songs("../project_train.csv")

    ans = test_methods(songs, 1000, "QDA")
    print("QDA correctness:", ans, "%")

    ans = test_methods(songs, 1000, "LDA")
    print("LDA correctness:", ans, "%")


if __name__ == "__main__":
    main()