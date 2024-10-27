from nearest_neighbour.weightedNN import classify

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


def label():
    training = get_songs("../project_train.csv")
    liked_rv, disliked_rv = generate_LDA_rvs(training)

    testing = []
    with open("../project_test.csv") as f:
        titles = f.readline().split()
        for line in f:
            songdata = [eval(x) for x in line.strip().split(",")]
            testing.append(songdata)

    for song in testing:
        ans = classify(song, liked_rv, disliked_rv)
        song.append(ans)


    with open("../project_test_labeled.csv", "w") as f:
        f.write("".join(titles+[",Label\n"]))
        for song in testing:
            line = ",".join([str(x) for x in song])+"\n"
            f.write(line)




def test():
    training = []
    with open("project_train.csv") as f:
        f.readline() # skip first
        for line in f:
            songdata = [eval(x) for x in line.strip().split(",")]
            training.append(Song(*songdata))

    testing = []
    with open("project_test_labeled.csv") as f:
        f.readline() # skip first
        for line in f:
            songdata = [eval(x) for x in line.strip().split(",")]
            testing.append(Song(*songdata))


    tot = 0
    for song in testing:
        ans = classify(song, training, 32, 1)
        if ans == song.Label:
            tot += 1

    print(tot, "of", len(testing))
