#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_dataset(books_path, ratings_path, users_path):
    """Analyze the Book-Crossings dataset and visualize filtering effects."""
    print("Loading datasets...")
    books_df = pd.read_csv(books_path, sep=',', on_bad_lines='skip', encoding='latin-1')
    ratings_df = pd.read_csv(ratings_path, sep=',', on_bad_lines='skip')
    users_df = pd.read_csv(users_path, sep=',', on_bad_lines='skip')
    
    print(f"\nOriginal dataset stats:")
    print(f"Books: {len(books_df)}")
    print(f"Ratings: {len(ratings_df)}")
    print(f"Users: {len(users_df)}")
    
    # Books with ratings
    books_with_ratings = ratings_df['ISBN'].unique()
    books_with_ratings_count = len(books_with_ratings)
    books_without_ratings = len(books_df) - books_with_ratings_count
    
    print(f"\nBooks with at least one rating: {books_with_ratings_count} ({books_with_ratings_count / len(books_df) * 100:.2f}%)")
    print(f"Books without ratings: {books_without_ratings} ({books_without_ratings / len(books_df) * 100:.2f}%)")
    
    # Convert ratings to explicit feedback (ratings 0 are implicit and will be removed)
    explicit_ratings_df = ratings_df[ratings_df['Book-Rating'] > 0]
    
    print(f"\nExplicit ratings (> 0): {len(explicit_ratings_df)} ({len(explicit_ratings_df) / len(ratings_df) * 100:.2f}%)")
    print(f"Implicit ratings (= 0): {len(ratings_df) - len(explicit_ratings_df)} ({(len(ratings_df) - len(explicit_ratings_df)) / len(ratings_df) * 100:.2f}%)")
    
    # Analyze book popularity (number of ratings per book)
    book_review_counts = explicit_ratings_df['ISBN'].value_counts()
    
    # Calculate books with different thresholds
    books_with_10plus_reviews = len(book_review_counts[book_review_counts >= 10])
    books_with_5to9_reviews = len(book_review_counts[(book_review_counts >= 5) & (book_review_counts < 10)])
    books_with_3to4_reviews = len(book_review_counts[(book_review_counts >= 3) & (book_review_counts < 5)])
    books_with_1to2_reviews = len(book_review_counts[(book_review_counts >= 1) & (book_review_counts < 3)])
    
    print(f"\nBooks with 10+ reviews: {books_with_10plus_reviews} ({books_with_10plus_reviews / books_with_ratings_count * 100:.2f}%)")
    print(f"Books with 5-9 reviews: {books_with_5to9_reviews} ({books_with_5to9_reviews / books_with_ratings_count * 100:.2f}%)")
    print(f"Books with 3-4 reviews: {books_with_3to4_reviews} ({books_with_3to4_reviews / books_with_ratings_count * 100:.2f}%)")
    print(f"Books with 1-2 reviews: {books_with_1to2_reviews} ({books_with_1to2_reviews / books_with_ratings_count * 100:.2f}%)")
    
    # Analyze user activity (number of ratings per user)
    user_review_counts = explicit_ratings_df['User-ID'].value_counts()
    
    # Calculate users with different thresholds
    users_with_10plus_reviews = len(user_review_counts[user_review_counts >= 10])
    users_with_5to9_reviews = len(user_review_counts[(user_review_counts >= 5) & (user_review_counts < 10)])
    users_with_3to4_reviews = len(user_review_counts[(user_review_counts >= 3) & (user_review_counts < 5)])
    users_with_1to2_reviews = len(user_review_counts[(user_review_counts >= 1) & (user_review_counts < 3)])
    
    total_users_with_ratings = len(user_review_counts)
    
    print(f"\nUsers with 10+ reviews: {users_with_10plus_reviews} ({users_with_10plus_reviews / total_users_with_ratings * 100:.2f}%)")
    print(f"Users with 5-9 reviews: {users_with_5to9_reviews} ({users_with_5to9_reviews / total_users_with_ratings * 100:.2f}%)")
    print(f"Users with 3-4 reviews: {users_with_3to4_reviews} ({users_with_3to4_reviews / total_users_with_ratings * 100:.2f}%)")
    print(f"Users with 1-2 reviews: {users_with_1to2_reviews} ({users_with_1to2_reviews / total_users_with_ratings * 100:.2f}%)")
    
    # Calculate impact of filtering books with less than 10 reviews and users with less than 3 reviews
    popular_books = book_review_counts[book_review_counts >= 10].index
    filtered_ratings_by_books = explicit_ratings_df[explicit_ratings_df['ISBN'].isin(popular_books)]
    
    active_users = user_review_counts[user_review_counts >= 3].index
    filtered_ratings = filtered_ratings_by_books[filtered_ratings_by_books['User-ID'].isin(active_users)]
    
    remaining_books = len(filtered_ratings['ISBN'].unique())
    remaining_users = len(filtered_ratings['User-ID'].unique())
    remaining_ratings = len(filtered_ratings)
    
    print(f"\nAfter filtering (books with <10 reviews and users with <3 reviews):")
    print(f"Remaining books: {remaining_books} ({remaining_books / books_with_ratings_count * 100:.2f}% of books with ratings)")
    print(f"Remaining users: {remaining_users} ({remaining_users / total_users_with_ratings * 100:.2f}% of users with ratings)")
    print(f"Remaining ratings: {remaining_ratings} ({remaining_ratings / len(explicit_ratings_df) * 100:.2f}% of explicit ratings)")
    
    # Visualize book popularity distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(book_review_counts, log_scale=True)
    plt.title('Distribution of Book Popularity (log scale)')
    plt.xlabel('Number of Ratings per Book')
    plt.ylabel('Count')
    plt.axvline(x=10, color='r', linestyle='--', label='Threshold (10 reviews)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('book_popularity_distribution.png')
    plt.close()
    
    # Visualize user activity distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(user_review_counts, log_scale=True)
    plt.title('Distribution of User Activity (log scale)')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Count')
    plt.axvline(x=3, color='r', linestyle='--', label='Threshold (3 reviews)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('user_activity_distribution.png')
    plt.close()
    
    # Visualize rating distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(explicit_ratings_df['Book-Rating'], bins=10, kde=True)
    plt.title('Distribution of Book Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('rating_distribution.png')
    plt.close()
    
    # Visualize impact of filtering
    labels = ['Original', 'Explicit\nRatings', 'After\nFiltering']
    ratings_counts = [len(ratings_df), len(explicit_ratings_df), remaining_ratings]
    books_counts = [len(books_with_ratings), len(book_review_counts), remaining_books]
    users_counts = [len(users_df['User-ID'].unique()), total_users_with_ratings, remaining_users]
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    
    ax[0].bar(labels, ratings_counts, color='skyblue')
    ax[0].set_title('Number of Ratings')
    ax[0].set_ylabel('Count')
    for i, v in enumerate(ratings_counts):
        ax[0].text(i, v, f"{v:,}", ha='center', va='bottom')
    
    ax[1].bar(labels, books_counts, color='lightgreen')
    ax[1].set_title('Number of Books')
    for i, v in enumerate(books_counts):
        ax[1].text(i, v, f"{v:,}", ha='center', va='bottom')
    
    ax[2].bar(labels, users_counts, color='salmon')
    ax[2].set_title('Number of Users')
    for i, v in enumerate(users_counts):
        ax[2].text(i, v, f"{v:,}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('filtering_impact.png')
    plt.close()
    
    return {
        'original': {
            'books': len(books_df),
            'ratings': len(ratings_df),
            'users': len(users_df)
        },
        'filtered': {
            'books': remaining_books,
            'ratings': remaining_ratings,
            'users': remaining_users
        }
    }

if __name__ == "__main__":
    # Paths to dataset files
    books_path = 'Recommender/Books.csv'
    ratings_path = 'Recommender/Ratings.csv'
    users_path = 'Recommender/Users.csv'
    
    stats = analyze_dataset(books_path, ratings_path, users_path) 