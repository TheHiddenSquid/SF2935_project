import matplotlib.pyplot as plt
from typing import List
import random

# Avrage preformance: 75 % correctness (unseen data)
# Preformance: 80 % correctness (training data)

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


def guess_label(song, liked_pdfs, disliked_pdfs, p_liked, p_disliked, bins):
    if eval_pdf(song, liked_pdfs, bins) * p_liked > eval_pdf(song, disliked_pdfs, bins) * p_disliked:
        ans = 1
    else:
        ans = 0
    return ans


def test_precent_correct_on_dataset(songs, bins, no_tests, no_training_songs):
    if no_training_songs > 0.8 * len(songs):
        raise ValueError(r"Training can't be more than 80% of the data")

    count = 0
    for _ in range(no_tests):
        training_songs = random.sample(songs, no_training_songs)
        testing_songs = [x for x in songs if x not in training_songs]

        liked_pmfs, disliked_pmfs = generate_pmfs(training_songs, bins)
        
    
        # Hard to know irl (prior guess used from dataset)
        p_liked = sum(1 for x in training_songs if x.Label == 1) / len(training_songs)
        p_disliked = 1 - p_liked

        # Preform test with rest of dataset
        correct = 0
        tot = 0
        for song in testing_songs:
            guess = guess_label(song, liked_pmfs, disliked_pmfs, p_liked, p_disliked, bins)
            if guess == song.Label:
                correct += 1
            tot += 1
        
        count +=correct/tot

    return count/no_tests


def optimize_bins(songs, b_max, no_training_songs):
    stats = []
    bins = []
    for i in range(5, b_max):
        res = test_precent_correct_on_dataset(songs, bins=i, no_tests=500, no_training_songs=no_training_songs)
        stats.append(res)
        bins.append(i)

    plt.plot(bins, stats)
    plt.xlabel("Bins")
    plt.ylabel("correct %")
    plt.title("Optimize number of bins")
    plt.show()
    print("Best:")
    print(f"bins: {bins[stats.index(max(stats))]}, res: {max(stats)}")

    return bins[stats.index(max(stats))]


def main():
    random.seed("random")
    songs = get_songs("project_train.csv")
    
    ans = test_precent_correct_on_dataset(songs, bins=12, no_tests=200, no_training_songs=400)
    print(ans)



if __name__ == "__main__":
    main()