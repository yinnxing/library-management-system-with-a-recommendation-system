#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from book_recommender import BookRecommender

def create_sample_data(original_dir, sample_dir, sample_size=2000, random_seed=42):
    """Create smaller sample datasets for demo purposes."""
    np.random.seed(random_seed)
    
    # Create sample directory if it doesn't exist
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # Sample books
    books_df = pd.read_csv(os.path.join(original_dir, "Books.csv"), sep=',', on_bad_lines='skip', encoding='latin-1')
    sampled_books = books_df.sample(min(sample_size, len(books_df)), random_state=random_seed)
    sampled_isbns = set(sampled_books['ISBN'])
    
    # Sample users and their ratings for the sampled books
    ratings_df = pd.read_csv(os.path.join(original_dir, "Ratings.csv"), sep=',', on_bad_lines='skip')
    sampled_ratings = ratings_df[ratings_df['ISBN'].isin(sampled_isbns)]
    
    # Get users who rated these books
    sampled_user_ids = set(sampled_ratings['User-ID'])
    
    # Get all ratings by these users to preserve their rating history
    sampled_ratings = ratings_df[ratings_df['User-ID'].isin(sampled_user_ids)]
    
    # Update isbns to include all books rated by sampled users
    sampled_isbns = set(sampled_ratings['ISBN'])
    sampled_books = books_df[books_df['ISBN'].isin(sampled_isbns)]
    
    # Sample users
    users_df = pd.read_csv(os.path.join(original_dir, "Users.csv"), sep=',', on_bad_lines='skip')
    sampled_users = users_df[users_df['User-ID'].isin(sampled_user_ids)]
    
    # Save sampled data
    sampled_books.to_csv(os.path.join(sample_dir, "Books.csv"), index=False)
    sampled_ratings.to_csv(os.path.join(sample_dir, "Ratings.csv"), index=False)
    sampled_users.to_csv(os.path.join(sample_dir, "Users.csv"), index=False)
    
    print(f"Created sample data with {len(sampled_books)} books, {len(sampled_ratings)} ratings, and {len(sampled_users)} users.")
    return sampled_books, sampled_ratings, sampled_users

def run_demo():
    original_dir = "Recommender"
    sample_dir = "Recommender/sample"
    
    # Create sample data if it doesn't exist yet
    if not os.path.exists(os.path.join(sample_dir, "Books.csv")):
        sampled_books, sampled_ratings, sampled_users = create_sample_data(original_dir, sample_dir)
    
    # Create and initialize recommender with sample data
    recommender = BookRecommender(
        books_path=os.path.join(sample_dir, "Books.csv"),
        ratings_path=os.path.join(sample_dir, "Ratings.csv"),
        users_path=os.path.join(sample_dir, "Users.csv")
    )
    
    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    recommender.load_data()
    
    # Train models
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    print("\nTraining content-based model...")
    # Use a smaller max_books value for the demo to avoid memory issues
    recommender.train_content_based_model(max_books=1000)
    
    print("\nTraining collaborative filtering model...")
    recommender.train_collaborative_filtering()
    
    # Evaluate models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    print("\nEvaluating collaborative filtering model:")
    recommender.evaluate_collaborative_filtering()
    
    print("\nEvaluating at different k values:")
    recommender.evaluate_metrics_at_k(k_values=[5, 10, 15])
    
    # Find an active user with many ratings
    active_users = recommender.ratings_df['User-ID'].value_counts()
    active_user_id = active_users.index[0]
    
    # Get recommendations
    print("\n" + "="*50)
    print(f"RECOMMENDATIONS FOR USER {active_user_id}")
    print("="*50)
    
    # Get collaborative filtering recommendations
    print("\nCollaborative Filtering Recommendations:")
    cf_recs = recommender.get_collaborative_recommendations(active_user_id, n=5)
    print(cf_recs[['ISBN', 'Book-Title', 'Book-Author']])
    
    # Get hybrid recommendations
    print("\nHybrid Recommendations:")
    hybrid_recs = recommender.get_hybrid_recommendations(active_user_id, n=5)
    print(hybrid_recs[['ISBN', 'Book-Title', 'Book-Author']])
    
    # Show user's past ratings
    user_ratings = recommender.ratings_df[recommender.ratings_df['User-ID'] == active_user_id]
    user_books = recommender.books_df[recommender.books_df['ISBN'].isin(user_ratings['ISBN'])]
    merged_ratings = pd.merge(user_ratings, user_books, on='ISBN')
    
    print(f"\nUser {active_user_id}'s Top 5 Rated Books:")
    top_books = merged_ratings.sort_values('Book-Rating', ascending=False).head(5)
    print(top_books[['Book-Title', 'Book-Author', 'Book-Rating']])
    
    # Find a popular book
    book_popularity = recommender.ratings_df['ISBN'].value_counts()
    popular_isbn = book_popularity.index[0]
    popular_book = recommender.books_df[recommender.books_df['ISBN'] == popular_isbn].iloc[0]
    
    print("\n" + "="*50)
    print(f"SIMILAR BOOKS TO \"{popular_book['Book-Title']}\" by {popular_book['Book-Author']}")
    print("="*50)
    
    # Get content-based recommendations
    book_index = recommender.books_df[recommender.books_df['ISBN'] == popular_isbn].index[0]
    content_recs = recommender.get_content_based_recommendations(book_index, n=5)
    
    print("\nContent-Based Recommendations:")
    print(content_recs[['ISBN', 'Book-Title', 'Book-Author']])

if __name__ == "__main__":
    run_demo() 