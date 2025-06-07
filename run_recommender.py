#!/usr/bin/env python3
import sys
import pandas as pd
import argparse
from book_recommender import BookRecommender

def main():
    parser = argparse.ArgumentParser(description='Book Recommendation System')
    parser.add_argument('--train', action='store_true', help='Train the recommendation system')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the recommendation system')
    parser.add_argument('--optimize', action='store_true', help='Optimize hybrid weight')
    parser.add_argument('--recommend', action='store_true', help='Get recommendations')
    parser.add_argument('--user-id', type=int, help='User ID to get recommendations for')
    parser.add_argument('--book-isbn', type=str, help='Book ISBN to get content-based recommendations for')
    parser.add_argument('--top-n', type=int, default=10, help='Number of recommendations to return')
    parser.add_argument('--algorithm', type=str, default='svd', choices=['svd', 'knn'], 
                        help='Collaborative filtering algorithm to use')
    parser.add_argument('--k-values', type=str, default='5,10,15,20', 
                        help='Comma-separated list of k values for evaluation')
    parser.add_argument('--max-books', type=int, default=5000, 
                        help='Maximum number of books to include in content-based model (to avoid memory issues)')
    
    args = parser.parse_args()
    
    # Create and load recommender system
    recommender = BookRecommender(
        books_path='Recommender/Books.csv',
        ratings_path='Recommender/Ratings.csv',
        users_path='Recommender/Users.csv'
    )
    
    print("Loading data...")
    recommender.load_data()
    
    # Train models if needed
    if args.train or args.evaluate or args.optimize or args.recommend:
        print("Training content-based model...")
        recommender.train_content_based_model(max_books=args.max_books)
        
        print("Training collaborative filtering model...")
        recommender.train_collaborative_filtering(algorithm=args.algorithm)
    
    # Evaluate models if requested
    if args.evaluate:
        print("\nEvaluating collaborative filtering model:")
        recommender.evaluate_collaborative_filtering()
        
        k_values = [int(k) for k in args.k_values.split(',')]
        print(f"\nEvaluating metrics at k values: {k_values}")
        metrics = recommender.evaluate_metrics_at_k(k_values=k_values)
        
        # Find optimal k value based on F1 score
        best_k = max(metrics.items(), key=lambda x: x[1]['f1'])[0]
        print(f"\nBest k value: {best_k} with F1 score: {metrics[best_k]['f1']:.4f}")
    
    # Optimize hybrid weight if requested
    if args.optimize:
        print("\nOptimizing hybrid weight:")
        optimal_weight, weight_results = recommender.optimize_hybrid_weight(user_sample=50, k=10)
        print(f"Optimal weight for hybrid model: {optimal_weight}")
    
    # Get recommendations if requested
    if args.recommend:
        if args.user_id is not None:
            user_id = args.user_id
            top_n = args.top_n
            
            # Check if user exists in the dataset
            if user_id not in recommender.ratings_df['User-ID'].unique():
                print(f"User {user_id} not found in the dataset.")
                # Get a random user
                user_id = recommender.ratings_df['User-ID'].value_counts().index[0]
                print(f"Using random user {user_id} instead.")
            
            # Get collaborative filtering recommendations
            print(f"\nCollaborative filtering recommendations for user {user_id}:")
            cf_recs = recommender.get_collaborative_recommendations(user_id, n=top_n)
            print(cf_recs[['ISBN', 'Book-Title', 'Book-Author']])
            
            # Get hybrid recommendations
            print(f"\nHybrid recommendations for user {user_id}:")
            hybrid_recs = recommender.get_hybrid_recommendations(user_id, n=top_n)
            print(hybrid_recs[['ISBN', 'Book-Title', 'Book-Author']])
            
            # Display user's past ratings for context
            user_ratings = recommender.ratings_df[recommender.ratings_df['User-ID'] == user_id]
            user_rated_books = recommender.books_df[recommender.books_df['ISBN'].isin(user_ratings['ISBN'])]
            merged_ratings = pd.merge(user_ratings, user_rated_books, on='ISBN')
            
            print(f"\nUser {user_id}'s past ratings:")
            print(merged_ratings[['Book-Title', 'Book-Author', 'Book-Rating']].sort_values('Book-Rating', ascending=False))
        
        elif args.book_isbn is not None:
            isbn = args.book_isbn
            top_n = args.top_n
            
            # Check if book exists in the dataset
            if isbn not in recommender.books_df['ISBN'].values:
                print(f"Book with ISBN {isbn} not found in the dataset.")
                # Get a random popular book
                popular_isbn = recommender.ratings_df['ISBN'].value_counts().index[0]
                isbn = popular_isbn
                print(f"Using random book with ISBN {isbn} instead.")
            
            # Get book info
            book_info = recommender.books_df[recommender.books_df['ISBN'] == isbn]
            print(f"\nFinding similar books to: {book_info['Book-Title'].values[0]} by {book_info['Book-Author'].values[0]}")
            
            # Get content-based recommendations
            book_index = recommender.books_df[recommender.books_df['ISBN'] == isbn].index[0]
            content_recs = recommender.get_content_based_recommendations(book_index, n=top_n)
            
            print(f"\nContent-based recommendations for book {isbn}:")
            print(content_recs[['ISBN', 'Book-Title', 'Book-Author']])
        
        else:
            print("Please provide either --user-id or --book-isbn for recommendations.")
            parser.print_help()
    
    # If no action is specified, print help
    if not (args.train or args.evaluate or args.optimize or args.recommend):
        parser.print_help()

if __name__ == "__main__":
    main() 