from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load movie dataset
movies_data = pd.read_csv("movies.csv")

# Select relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features into a single text
movies_data['combined_features'] = movies_data[selected_features].apply(lambda x: ' '.join(x), axis=1)

# Convert text to numerical data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

# Compute similarity matrix
similarity = cosine_similarity(feature_vectors)

@app.route('/api/recommend', methods=['GET'])
def recommend_movies():
    movie_name = request.args.get("title", "").strip()
    
    # Find closest matching movie
    list_of_titles = movies_data['title'].tolist()
    close_match = difflib.get_close_matches(movie_name, list_of_titles, n=1)
    
    if not close_match:
        return jsonify({"recommendations": []})  # Return empty if not found

    close_match = close_match[0]
    index_of_movie = movies_data[movies_data.title == close_match].index[0]

    # Retrieve similarity scores
    similarity_scores = list(enumerate(similarity[index_of_movie]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]

    # Get recommended movie titles
    recommended_movies = [movies_data.iloc[movie[0]].title for movie in sorted_similar_movies]

    return jsonify({"recommendations": recommended_movies})

if __name__ == '__main__':
    app.run(debug=True)
