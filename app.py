import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD

# Set your TMDB API key here
TMDB_API_KEY = "4150bd5ca9e960f527d69c5701045bc0"

@st.cache_data(show_spinner=False)
def load_data():
    with open("best_svd.pkl", "rb") as f:
        best_svd = pickle.load(f)
    with open("movies.pkl", "rb") as f:
        movies = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("movie_indices.pkl", "rb") as f:
        movie_indices = pickle.load(f)
    with open("ratings.pkl", "rb") as f:
        ratings = pickle.load(f)
    return best_svd, movies, tfidf_matrix, movie_indices, ratings

best_svd, movies, tfidf_matrix, movie_indices, ratings = load_data()

def get_movie_poster(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    poster_path = data.get("poster_path")
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

def get_similar_movies(movie_title, n=10):
    if movie_title not in movie_indices:
        return pd.DataFrame(columns=["movieId", "title", "tmdbId"])
    idx = movie_indices[movie_title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = np.argsort(sim_scores)[::-1][1:n+1]
    return movies.iloc[similar_idx][["movieId", "title", "tmdbId"]]

def get_svd_predictions(user_id, movie_ids):
    preds = []
    for movie_id in movie_ids:
        pred = best_svd.predict(user_id, movie_id, 0)
        preds.append({"movieId": movie_id, "predicted_rating": pred.est})
    return pd.DataFrame(preds)

def get_hybrid_recommendations(user_id, movie_title, alpha=0.5, num_recs=10):
    content_recs = get_similar_movies(movie_title, num_recs)
    if content_recs.empty:
        return pd.DataFrame(columns=["movieId", "title", "tmdbId", "final_score"])
    svd_preds = get_svd_predictions(user_id, content_recs["movieId"].tolist())
    hybrid_df = content_recs.merge(svd_preds, on="movieId", how="left")
    avg_rating = svd_preds["predicted_rating"].mean() if not svd_preds.empty else 3.0
    hybrid_df["predicted_rating"] = hybrid_df["predicted_rating"].fillna(avg_rating)
    hybrid_df["content_score"] = np.linspace(1, 0, len(hybrid_df))
    hybrid_df["final_score"] = alpha * hybrid_df["predicted_rating"] + (1 - alpha) * hybrid_df["content_score"]
    return hybrid_df.sort_values(by="final_score", ascending=False).head(num_recs)

# Streamlit UI
st.title("Hybrid Movie Recommendation")

user_id = st.sidebar.number_input("User ID", min_value=1, value=1)
movie_list = sorted(movies["title"].unique())
movie_title = st.sidebar.selectbox("Select Movie", movie_list)
alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.5)
num_recs = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

if st.sidebar.button("Get Recommendations"):
    recommendations = get_hybrid_recommendations(user_id, movie_title, alpha, num_recs)
    if recommendations.empty:
        st.write("No recommendations found. Please try another movie.")
    else:
        for _, row in recommendations.iterrows():
            st.markdown(f"### {row['title']}  \nFinal Score: {row['final_score']:.2f}")
            poster = get_movie_poster(row["tmdbId"])
            if poster:
                st.image(poster, width=200)
            else:
                st.write("Poster not available.")
            st.markdown("---")
