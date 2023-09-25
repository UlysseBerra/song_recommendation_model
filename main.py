import pandas
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("dataset.csv")

for column in df.columns:
	df[column] = df[column].map(str)


def combine_features(row):
	return row['artists'] + "  " + row['popularity'] + "  " + row['danceability'] + "  " + row['energy'] + "  " + row['loudness'] + "  " + row['speechiness'] + "  " + row['acousticness'] + "  " + row['instrumentalness'] + "  " + row['liveness'] + "  " + row['valence'] + "  " + row['tempo'] + "  " + row['track_genre']


df["combined_features"] = df.apply(combine_features, axis=1)

cv = CountVectorizer()

count_matrix = df["combined_features"]

print(cosine_similarity(count_matrix))