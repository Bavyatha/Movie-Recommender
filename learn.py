import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies_data = pd.read_csv("movies.csv")  # Ensure movies.csv is in the same directory

# Display the first few rows
print("Dataset Loaded Successfully!")
print(movies_data.head())

# Select relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Handle missing values by replacing NaN with an empty string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features into a single text
movies_data['combined_features'] = movies_data[selected_features].apply(lambda x: ' '.join(x), axis=1)

# Convert text data into numerical data using TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

# Compute the similarity matrix
similarity = cosine_similarity(feature_vectors)

def recommend_movies(movie_name):
    # Get the list of all movie titles
    list_of_titles = movies_data['title'].tolist()
    
    # Find the closest matching movie name
    close_match = difflib.get_close_matches(movie_name, list_of_titles, n=1)
    
    if not close_match:
        print("Movie not found. Please check the spelling and try again.")
        return
    
    close_match = close_match[0]

    # Get index of the movie
    index_of_movie = movies_data[movies_data.title == close_match].index[0]

    # Retrieve similarity scores
    similarity_scores = list(enumerate(similarity[index_of_movie]))

    # Sort movies based on similarity score (descending order)
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]

    # Display recommended movies
    print("\nMovies recommended for you based on", close_match, ":")
    for movie in sorted_similar_movies:
        print(movies_data.iloc[movie[0]].title)

# Take user input for movie name
movie_name = input("\nEnter a movie name: ")
recommend_movies(movie_name)


