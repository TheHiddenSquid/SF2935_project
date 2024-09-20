from scipy import stats
import random

# Avrage preformance 75 %

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


def generate_pdfs(songs):
    liked_songs = [x for x in songs if x.Label == 1]
    disliked_songs = [x for x in songs if x.Label == 0]
    
    liked_pdfs = {}
    disliked_pdfs = {}

    # Beta distribution for all these:
    attributes = ["danceability", "energy", "speechiness", "acousticness", "liveness", "instrumentalness"]
    bounds_dict = {"danceability":[0,1], "energy":[0,1],"loudness":[-60,0], "speechiness":[0,1], "acousticness":[0,1], "liveness":[0,1], "valence":[0,1], "tempo":[20,200], "instrumentalness":[0,1]}

    for attr in attributes:
        m = bounds_dict[attr][0]    # For rescaling to [0,1]
        M = bounds_dict[attr][1]    # For rescaling to [0,1]
        liked_data = [(getattr(x, attr)-m)/(M-m) for x in liked_songs]
        disliked_data = [(getattr(x, attr)-m)/(M-m) for x in disliked_songs]

        a,b, *args = stats.fit(stats.beta, liked_data, [(0, 100), (0, 100)]).params
        liked_pdfs[attr] = stats.beta(a, b)
       
        a,b, *args = stats.fit(stats.beta, disliked_data, [(0, 100), (0, 100)]).params
        disliked_pdfs[attr] = stats.beta(a, b)
    

    # Kernel density estimation for these:
    attributes = ["loudness", "valence", "tempo"]

    for attr in attributes:
        m = bounds_dict[attr][0]    # For rescaling to [0,1]
        M = bounds_dict[attr][1]    # For rescaling to [0,1]
        liked_data = [(getattr(x, attr)-m)/(M-m) for x in liked_songs]
        disliked_data = [(getattr(x, attr)-m)/(M-m) for x in disliked_songs]

        liked_pdfs[attr] = stats.gaussian_kde(liked_data)
        disliked_pdfs[attr] = stats.gaussian_kde(disliked_data)


    # Categotical for "mode"
    attr = "mode"
    liked_data = [getattr(x, attr) for x in liked_songs]
    disliked_data = [getattr(x, attr) for x in disliked_songs]

    liked_pdfs[attr] = [liked_data.count(x)/len(liked_data) for x in [0,1]]
    disliked_pdfs[attr] = [disliked_data.count(x)/len(disliked_data) for x in [0,1]]


    # Categotical for "key"
    attr = "key"
    liked_data = [getattr(x, attr) for x in liked_songs]
    disliked_data = [getattr(x, attr) for x in disliked_songs]

    liked_pdfs[attr] = [liked_data.count(x)/len(liked_data) for x in range(12)]
    disliked_pdfs[attr] = [disliked_data.count(x)/len(disliked_data) for x in range(12)]
    

    return liked_pdfs, disliked_pdfs


def eval_pdf(song, pdfs, epsilon=0.01):
    tot = 1

    # Count all beta distributions:
    attributes = ["danceability", "energy", "speechiness", "acousticness", "liveness"]
    bounds_dict = {"danceability":[0,1], "energy":[0,1], "loudness":[-60,0], "speechiness":[0,1], "acousticness":[0,1], "liveness":[0,1], "valence":[0,1], "tempo":[20,200]}

    for attr in attributes:
        m = bounds_dict[attr][0]
        M = bounds_dict[attr][1]
        rel_x = (getattr(song, attr)-m)/(M-m)
        if rel_x - epsilon/2 < 0:
            tot *= pdfs[attr].cdf(epsilon)
        elif rel_x + epsilon/2 > 1:
            tot *= 1 - pdfs[attr].cdf(1-epsilon)
        else:
            tot *= pdfs[attr].cdf(rel_x+epsilon/2) - pdfs[attr].cdf(rel_x-epsilon/2)


    # Count the Kernel density estimations:
    attributes = ["loudness","valence","tempo"]

    for attr in attributes:
        m = bounds_dict[attr][0]
        M = bounds_dict[attr][1]
        rel_x = (getattr(song, attr)-m)/(M-m)
        if rel_x - epsilon/2 < 0:
            tot *= pdfs[attr].integrate_box_1d(0, rel_x+epsilon)
        elif rel_x + epsilon/2 > 1:
            tot *= pdfs[attr].integrate_box_1d(rel_x-epsilon, 1)
        else:
            tot *= pdfs[attr].integrate_box_1d(rel_x-epsilon/2, rel_x+epsilon/2)


    # Count the categorical-distributions:
    attr = "mode"
    tot *= pdfs[attr][getattr(song, attr)]

    attr = "key"
    tot *= pdfs[attr][getattr(song, attr)]

    #Custom for instrumentalness:
    attr = "instrumentalness"
    rel_x = getattr(song, attr)
    if rel_x < 0.01:
        rel_x = 0.01
    tot *= pdfs[attr].pdf(rel_x)

    return tot


def guess_label(song, liked_pdfs, disliked_pdfs, p_liked, p_disliked):
    if eval_pdf(song, liked_pdfs) * p_liked > eval_pdf(song, disliked_pdfs) * p_disliked:
        ans = 1
    else:
        ans = 0
    return ans


def test(songs, no_tests):
    count = 0
    for i in range(no_tests):
        print(i)
        testing_songs = random.sample(songs, 100)
        trainging_songs = [x for x in songs if x not in testing_songs]

        liked_pdfs, disliked_pdfs = generate_pdfs(trainging_songs)
        
        # Hard to know irl (prior guess used from dataset)
        p_liked = sum(1 for x in trainging_songs if x.Label == 1) / len(trainging_songs)
        p_disliked = 1 - p_liked

        # Preform test with rest of dataset
        correct = 0
        tot = 0
        for song in testing_songs:
            guess = guess_label(song, liked_pdfs, disliked_pdfs, p_liked, p_disliked)
            if guess == song.Label:
                correct += 1
            tot += 1
        
        count +=correct/tot

    return f"average percentage: {count/no_tests}"


def main():
    random.seed("ra1ndom")
    songs = get_songs()
   
    print(test(songs, no_tests=25))
   

if __name__ == "__main__":
    main()
