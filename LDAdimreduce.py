import numpy as np
from typing import List
import matplotlib.pyplot as plt

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


def Sw(data):
    liked_songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in data if song[-1] == 1], axis=1)
    liked_mean = np.transpose(np.array(np.mean(liked_songs_array, axis=1), ndmin = 2))
    disliked_songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in data if song[-1] == 0], axis=1)
    disliked_mean = np.transpose(np.array(np.mean(disliked_songs_array, axis=1), ndmin = 2))
    

    sum = np.zeros((11,11))

    for song in data:
        label = song[-1]
        vec = np.transpose(np.array(song[:11], ndmin = 2))

        if label == 1:
            sum += np.matmul(vec - liked_mean, np.transpose(vec - liked_mean))
        else:
            sum += np.matmul(vec - disliked_mean, np.transpose(vec - disliked_mean))
    return sum


def Sb(data):
    songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in data], axis=1)
    total_mean = np.transpose(np.array(np.mean(songs_array, axis=1), ndmin = 2))

    
    liked_songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in data if song[-1] == 1], axis=1)
    liked_mean = np.transpose(np.array(np.mean(liked_songs_array, axis=1), ndmin = 2))
    N1 = liked_songs_array.shape[1]

    disliked_songs_array = np.concatenate([np.array(song[:11], ndmin=2).transpose() for song in data if song[-1] == 0], axis=1)
    disliked_mean = np.transpose(np.array(np.mean(disliked_songs_array, axis=1), ndmin = 2))
    N0 = disliked_songs_array.shape[1]
   
 
    a = N1 * np.matmul(liked_mean-total_mean, np.transpose(liked_mean-total_mean))
    b = N0 * np.matmul(disliked_mean-total_mean, np.transpose(disliked_mean-total_mean))
    return a+b



def main():
    songs = get_songs("project_train.csv")

    sw = Sw(songs)
    sb = Sb(songs)

    matrix = np.linalg.inv(sw) * sb


    W1 = np.array(np.linalg.eig(matrix)[1][0], ndmin=2)
    W2 = np.array(np.linalg.eig(matrix)[1][1], ndmin=2)

    W = np.concatenate((W1, W2), axis = 0)


    new_liked = []
    new_disliked = []
    for song in songs:
        if song[-1] == 1:
            new_liked.append(np.matmul(W, np.transpose(np.array(song[:11], ndmin=2))))
        else:
            new_disliked.append(np.matmul(W, np.transpose(np.array(song[:11], ndmin=2))))



    plt.scatter([x[0] for x in new_liked], [x[1] for x in new_liked], label="liked")
    plt.scatter([x[0] for x in new_disliked], [x[1] for x in new_disliked], alpha=0.7, label="disliked")
    plt.legend(loc="upper right")
    plt.xlabel(r"$d_1$")
    plt.ylabel(r"$d_2$")
    plt.title("LDA dimensionality reduction")
    plt.show()


if __name__ == "__main__":
    main()