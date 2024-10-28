# Repo for SF2935 project

The project is about creating classification-algorithms for music. We have received a dataset of about 500 songs with audio features from Spotify's API and labels based on whether the song is liked or disliked by the TA.

The classification-algorithms use all 11 features, but to visually describe the dataset LDA has been used to project the data onto the 2D plane that makes the data as separable as possible. Here is that plane:
![alt text](https://github.com/TheHiddenSquid/SF2935_project/blob/main/images/LDA_2d.png?raw=true)

Of all the classifiers LDA was the most successful one. It classified data correctly about 79.8% of the time.
