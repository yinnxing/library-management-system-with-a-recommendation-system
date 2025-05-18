import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import re
import pickle
import os

class HybridBookRecommender:
    def __init__(self, content_weight=0.3, collab_weight=0.5, popular_weight=0.2, optimize_weights=False):
        """
        Initialize the hybrid recommender system with component weights.
        
        Args:
            content_weight: Weight for content-based recommendations
            collab_weight: Weight for collaborative filtering recommendations
            popular_weight: Weight for popularity-based recommendations
            optimize_weights: Whether to optimize weights using validation data
        """
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.popular_weight = popular_weight
        self.optimize_weights = optimize_weights
        
        self.books_df = None
        self.ratings_df = None
        self.users_df = None
        self.merged_df = None
        self.user_book_matrix = None
        self.item_similarity_matrix = None
        self.book_content_matrix = None
        self.popular_books = None
        self.svd_model = None
        
        # For evaluation and cold start handling
        self.train_data = None
        self.test_data = None
        self.cold_start_threshold = 5
        self.book_popularity_map = {}
        self.user_mean_ratings = {}
        self.global_mean_rating = 0
        
    def load_data(self, books_path, ratings_path, users_path):
        """Load the book dataset files"""
        self.books_df = pd.read_csv(books_path, low_memory=False)
        self.ratings_df = pd.read_csv(ratings_path, low_memory=False)
        self.users_df = pd.read_csv(users_path, low_memory=False)
        
        print(f"Loaded {len(self.books_df)} books, {len(self.ratings_df)} ratings, and {len(self.users_df)} users")
        
    def preprocess_data(self, min_book_ratings=15, min_user_ratings=3, verbose=True):
        """
        Preprocess and clean the data with comprehensive steps:
        1. Data merging and initial cleaning
        2. Data type conversion and normalization
        3. Feature engineering
        4. Handling missing values
        5. Filtering based on frequency thresholds
        6. Creating matrices and data structures
        
        Args:
            min_book_ratings: Minimum ratings for a book to be included
            min_user_ratings: Minimum ratings for a user to be included
            verbose: Whether to print detailed progress
        """
        if verbose:
            print("Step 1/6: Merging datasets...")
        # Merge datasets
        df1 = self.books_df.merge(self.ratings_df, how="left", on="ISBN")
        self.merged_df = df1.merge(self.users_df, how="left", on="User-ID")
        
        # Drop rows with missing User-ID or Book-Rating
        initial_size = len(self.merged_df)
        self.merged_df = self.merged_df.dropna(subset=['User-ID', 'Book-Rating'])
        if verbose:
            print(f"  - Dropped {initial_size - len(self.merged_df)} rows with missing User-ID or Book-Rating")
            
        if verbose:
            print("Step 2/6: Cleaning and converting data types...")
        # Clean author names
        self.merged_df["Book-Author"] = self.merged_df["Book-Author"].astype(str)
        self.merged_df["Book-Author"] = self.merged_df["Book-Author"].str.strip()
        self.merged_df["Book-Author"] = self.merged_df["Book-Author"].str.replace(r'\s+', ' ', regex=True)
        self.merged_df["Book-Author"] = self.merged_df["Book-Author"].str.title()
        
        # Clean book titles
        self.merged_df["Book-Title"] = self.merged_df["Book-Title"].astype(str)
        self.merged_df["Book-Title"] = self.merged_df["Book-Title"].str.strip()
        
        # Clean publisher names
        self.merged_df["Publisher"] = self.merged_df["Publisher"].astype(str) 
        self.merged_df["Publisher"] = self.merged_df["Publisher"].str.strip()
        self.merged_df["Publisher"] = self.merged_df["Publisher"].str.replace(r'\s+', ' ', regex=True)
        
        # Convert datatypes with safer approach
        self.merged_df['User-ID'] = self.merged_df['User-ID'].astype('int')
        
        # Handle Year-Of-Publication safely
        self.merged_df['Year-Of-Publication'] = pd.to_numeric(self.merged_df['Year-Of-Publication'], errors='coerce')
        # Replace invalid years (too old or future) with NaN
        self.merged_df.loc[self.merged_df['Year-Of-Publication'] < 1800, 'Year-Of-Publication'] = None
        self.merged_df.loc[self.merged_df['Year-Of-Publication'] > 2023, 'Year-Of-Publication'] = None
        
        # Handle Book-Rating - ensure numeric
        self.merged_df['Book-Rating'] = pd.to_numeric(self.merged_df['Book-Rating'], errors='coerce')
        
        # Per dataset documentation: ratings are either explicit (1-10) or implicit (0)
        # No need to rescale, but ensure we understand the distribution
        rating_counts = self.merged_df['Book-Rating'].value_counts().sort_index()
        if verbose:
            print("  - Rating distribution:")
            print(f"    Implicit ratings (0): {rating_counts.get(0, 0)}")
            explicit_ratings = rating_counts[rating_counts.index > 0]
            for rating, count in explicit_ratings.items():
                print(f"    Rating {int(rating)}: {count}")
        
        # Remove unrated books (rating = 0, which are implicit ratings)
        initial_size = len(self.merged_df)
        self.merged_df = self.merged_df[self.merged_df["Book-Rating"] > 0]
        if verbose:
            print(f"  - Removed {initial_size - len(self.merged_df)} implicit ratings (rating=0)")
            print(f"  - Ratings range: {self.merged_df['Book-Rating'].min()}-{self.merged_df['Book-Rating'].max()}")
        
        if verbose:
            print("Step 3/6: Feature engineering...")
        # Extract publication decade as a feature
        self.merged_df['Decade'] = (self.merged_df['Year-Of-Publication'] // 10) * 10
        self.merged_df['Decade'] = self.merged_df['Decade'].fillna(2000)  # Default to 2000s for missing years
        
        # Create a "book age" feature (relative to 2023)
        current_year = 2023
        self.merged_df['Book-Age'] = current_year - self.merged_df['Year-Of-Publication']
        
        # Extract user location features
        if 'Location' in self.merged_df.columns:
            # Split location into city, state/province, country
            location_parts = self.merged_df['Location'].str.split(',', expand=True)
            if location_parts.shape[1] >= 3:
                self.merged_df['City'] = location_parts[0].str.strip()
                self.merged_df['State'] = location_parts[1].str.strip()
                self.merged_df['Country'] = location_parts[2].str.strip()
            elif location_parts.shape[1] == 2:
                self.merged_df['City'] = location_parts[0].str.strip()
                self.merged_df['Country'] = location_parts[1].str.strip()
                
        if verbose:
            print("Step 4/6: Handling missing values...")
        # Fill missing values in key fields
        # For publication year, use median year
        median_year = self.merged_df['Year-Of-Publication'].median()
        self.merged_df['Year-Of-Publication'] = self.merged_df['Year-Of-Publication'].fillna(median_year)
        
        # For missing authors or publishers, mark as unknown
        self.merged_df['Book-Author'] = self.merged_df['Book-Author'].fillna('Unknown Author')
        self.merged_df['Publisher'] = self.merged_df['Publisher'].fillna('Unknown Publisher')
        
        if verbose:
            missing_vals = self.merged_df[['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']].isna().sum()
            print(f"  - Missing values after handling: {missing_vals.to_dict()}")
        
        if verbose:
            print("Step 5/6: Filtering based on frequency thresholds...")
        # Filter out rarely rated books and users with few ratings
        book_counts = self.merged_df['Book-Title'].value_counts()
        user_counts = self.merged_df['User-ID'].value_counts()
        
        popular_books = book_counts[book_counts >= min_book_ratings].index
        active_users = user_counts[user_counts >= min_user_ratings].index
        
        initial_books = self.merged_df['Book-Title'].nunique()
        initial_users = self.merged_df['User-ID'].nunique()
        initial_ratings = len(self.merged_df)
        
        self.merged_df = self.merged_df[self.merged_df['Book-Title'].isin(popular_books)]
        self.merged_df = self.merged_df[self.merged_df['User-ID'].isin(active_users)]
        
        if verbose:
            print(f"  - Books: {initial_books} -> {self.merged_df['Book-Title'].nunique()} (kept {self.merged_df['Book-Title'].nunique()/initial_books:.1%})")
            print(f"  - Users: {initial_users} -> {self.merged_df['User-ID'].nunique()} (kept {self.merged_df['User-ID'].nunique()/initial_users:.1%})")
            print(f"  - Ratings: {initial_ratings} -> {len(self.merged_df)} (kept {len(self.merged_df)/initial_ratings:.1%})")
        
        if verbose:
            print("Step 6/6: Creating matrices and data structures...")
        # Create user-book matrix
        self.user_book_matrix = self.merged_df.pivot_table(
            index='User-ID', 
            columns='Book-Title',
            values='Book-Rating'
        ).fillna(0)
        
        # Calculate book popularity
        self.popular_books = self.merged_df.groupby('Book-Title')['Book-Rating'].agg(['count', 'mean'])
        self.popular_books['score'] = self.popular_books['count'] * self.popular_books['mean']
        self.popular_books = self.popular_books.sort_values('score', ascending=False)
        
        # Calculate global statistics
        self.global_mean_rating = self.merged_df['Book-Rating'].mean()
        self.rating_std = self.merged_df['Book-Rating'].std()
        
        # Calculate user statistics
        user_stats = self.merged_df.groupby('User-ID')['Book-Rating'].agg(['mean', 'std', 'count'])
        self.user_stats = user_stats
        
        # Calculate item statistics
        item_stats = self.merged_df.groupby('Book-Title')['Book-Rating'].agg(['mean', 'std', 'count'])
        self.item_stats = item_stats
        
        # Calculate sparsity
        total_possible_ratings = self.merged_df['User-ID'].nunique() * self.merged_df['Book-Title'].nunique()
        actual_ratings = len(self.merged_df)
        sparsity = 1 - (actual_ratings / total_possible_ratings)
        
        if verbose:
            print(f"\nPreprocessing complete!")
            print(f"Final dataset: {len(self.merged_df)} ratings from {self.merged_df['User-ID'].nunique()} users on {self.merged_df['Book-Title'].nunique()} books")
            print(f"Sparsity: {sparsity:.2%} (higher means more sparse)")
            print(f"Rating range: {self.merged_df['Book-Rating'].min()}-{self.merged_df['Book-Rating'].max()}, mean: {self.global_mean_rating:.2f}, std: {self.rating_std:.2f}")
            print(f"User-book matrix shape: {self.user_book_matrix.shape}")
            print(f"Most popular book: {self.popular_books.index[0]} ({self.popular_books['count'].iloc[0]} ratings, avg: {self.popular_books['mean'].iloc[0]:.2f})")
            
            # Print most prolific users
            top_users = user_stats.sort_values('count', ascending=False).head(1)
            print(f"Most active user: ID {top_users.index[0]} ({top_users['count'].iloc[0]} ratings, avg: {top_users['mean'].iloc[0]:.2f})")
        
        return self.merged_df
        
    def build_content_based_model(self, verbose=True):
        """
        Build the content-based filtering model with enhanced features
        
        This creates a richer content-based model using:
        1. Author information
        2. Publisher information
        3. Publication year and decade
        4. Genre/category extraction where possible
        5. Text features from titles
        """
        if verbose:
            print("Building content-based model...")
            print("Step 1/4: Preparing book content features")
            
        # Create book content features from the unique books in our dataset
        books_content = self.merged_df.drop_duplicates('Book-Title')[
            ['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication', 'Decade']
        ]
        
        # Create additional features if available
        if 'Category' in self.merged_df.columns:
            books_content = books_content.merge(
                self.merged_df.drop_duplicates('Book-Title')[['Book-Title', 'Category']], 
                on='Book-Title'
            )
        
        # Extract potential genre information from titles
        if verbose:
            print("Step 2/4: Extracting genre information from titles")
            
        # Common genre keywords that might appear in titles
        genre_keywords = [
            'mystery', 'romance', 'fantasy', 'science fiction', 'sci-fi', 'scifi', 
            'thriller', 'horror', 'biography', 'history', 'memoir', 'philosophy',
            'psychology', 'self-help', 'business', 'fiction', 'non-fiction',
            'adventure', 'young adult', 'children', 'poetry', 'drama', 'comedy',
            'dystopian', 'classic', 'historical', 'political', 'religious',
            'spiritual', 'travel', 'cookbook', 'guide', 'manual', 'reference'
        ]
        
        # Check titles for genre keywords
        for genre in genre_keywords:
            books_content[f'Genre_{genre.replace(" ", "_")}'] = books_content['Book-Title'].str.lower().str.contains(genre).astype(int)
            
        # Create decade indicator features
        if verbose:
            print("Step 3/4: Creating categorical features")
            
        # One-hot encode decade if there's enough variety
        decades = books_content['Decade'].dropna().unique()
        if len(decades) > 1:
            decade_dummies = pd.get_dummies(books_content['Decade'], prefix='decade')
            books_content = pd.concat([books_content, decade_dummies], axis=1)
        
        # Create a combined feature text with weighted components
        if verbose:
            print("Step 4/4: Creating TF-IDF content matrix")
            
        # Add more weight to author by repeating it
        books_content['content'] = (
            books_content['Book-Author'] + ' ' + 
            books_content['Book-Author'] + ' ' +  # Repeat author for more weight
            books_content['Publisher'] + ' ' + 
            books_content['Year-Of-Publication'].astype(str)
        )
        
        # Extract title words for additional features
        # Remove common stop words and punctuation
        books_content['title_words'] = books_content['Book-Title'].str.lower()\
            .str.replace(r'[^\w\s]', '', regex=True)\
            .str.replace(r'\d+', '', regex=True)
                
        # Combine title words into content
        books_content['content'] = books_content['content'] + ' ' + books_content['title_words']
        
        # For any additional categorical columns, add them to the content string
        for col in books_content.columns:
            if col.startswith('Genre_') and col in books_content:
                # Only add the genre term if it's present (value is 1)
                genre_term = col.replace('Genre_', '')
                books_content.loc[books_content[col] == 1, 'content'] += f' {genre_term}'
        
        # Create TF-IDF matrix with improved parameters
        tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            max_features=5000,    # Limit number of features to avoid overfitting
            min_df=2              # Ignore terms that appear in less than 2 documents
        )
        
        # Check if we have enough books for a meaningful TF-IDF
        if len(books_content) < 5:
            if verbose:
                print("Warning: Not enough unique books for meaningful content-based recommendations")
            # Create a simple similarity matrix based on other features
            self.book_content_matrix = pd.DataFrame(
                np.eye(len(books_content)),
                index=books_content['Book-Title'],
                columns=books_content['Book-Title']
            )
            return
            
        # Fit and transform
        try:
            tfidf_matrix = tfidf.fit_transform(books_content['content'])
            
            if verbose:
                print(f"  - Created TF-IDF matrix with {tfidf_matrix.shape[1]} features")
                
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create DataFrame from similarity matrix
            self.book_content_matrix = pd.DataFrame(
                similarity_matrix,
                index=books_content['Book-Title'],
                columns=books_content['Book-Title']
            )
            
            if verbose:
                print(f"Content-based model built successfully with {len(self.book_content_matrix)} books")
                # Show sample similarity for a random book
                sample_book = np.random.choice(self.book_content_matrix.index)
                similar_books = self.book_content_matrix[sample_book].sort_values(ascending=False)[1:6]
                print(f"\nSample similar books to '{sample_book}':")
                for book, score in similar_books.items():
                    print(f"  - {book} (similarity: {score:.2f})")
                    
        except Exception as e:
            print(f"Error building content-based model: {str(e)}")
            # Create an identity matrix as fallback
            self.book_content_matrix = pd.DataFrame(
                np.eye(len(books_content)),
                index=books_content['Book-Title'],
                columns=books_content['Book-Title']
            )
        
    def build_collaborative_model(self, n_components=100, verbose=True):
        """
        Build the collaborative filtering model using matrix factorization
        
        This enhanced version includes:
        1. Data normalization to remove user and item biases
        2. Optimized SVD component selection
        3. User-user and item-item similarity matrices
        4. Improved handling of sparse data
        
        Args:
            n_components: Number of latent factors for SVD decomposition
            verbose: Whether to print progress information
        """
        if verbose:
            print("Building collaborative filtering model...")
            print("Step 1/4: Preparing user-item matrix")
        
        # Check if we have enough data
        if self.user_book_matrix.shape[0] < 5 or self.user_book_matrix.shape[1] < 5:
            if verbose:
                print("Warning: Not enough data for meaningful collaborative filtering")
            # Create an identity matrix as fallback
            self.item_similarity_matrix = pd.DataFrame(
                np.eye(len(self.user_book_matrix.columns)),
                index=self.user_book_matrix.columns,
                columns=self.user_book_matrix.columns
            )
            return
        
        try:
            # For explicit ratings (1-10), center around the middle of the scale
            if verbose:
                print("Step 2/4: Normalizing ratings data")
                
            # Create a copy of the matrix for manipulation
            matrix_copy = self.user_book_matrix.copy()
            
            # Calculate mean and standard deviation of ratings (ignoring zeros)
            mean_rating = np.mean(matrix_copy.values[matrix_copy.values > 0])
            std_rating = np.std(matrix_copy.values[matrix_copy.values > 0])
            
            if verbose:
                print(f"  - Rating statistics: mean={mean_rating:.2f}, std={std_rating:.2f}")
                
            # Center the ratings around user means
            # Replace zeros with NaN to avoid affecting the means
            user_matrix = matrix_copy.replace(0, np.nan)
            
            # Calculate user means (only on movies they've rated)
            user_means = user_matrix.mean(axis=1)
            
            if verbose:
                print(f"  - User mean ratings: min={user_means.min():.2f}, max={user_means.max():.2f}, avg={user_means.mean():.2f}")
            
            # Center each user's ratings around their mean
            # This helps address user bias (some users rate everything high, others low)
            for user_id in user_matrix.index:
                # Get this user's mean rating
                user_mean = user_means[user_id]
                # Only adjust non-NaN values
                mask = ~np.isnan(user_matrix.loc[user_id])
                # Center around user mean
                user_matrix.loc[user_id, mask] = user_matrix.loc[user_id, mask] - user_mean
            
            # Replace NaN with 0 for computation (after centering)
            user_matrix = user_matrix.fillna(0)
            
            # Use this centered matrix for SVD
            if verbose:
                print("Step 3/4: Applying SVD decomposition")
                
            # Determine optimal number of components
            max_possible = min(user_matrix.shape[0], user_matrix.shape[1], n_components)
            actual_components = min(max_possible, n_components)
            
            if verbose and actual_components < n_components:
                print(f"  - Reducing components from {n_components} to {actual_components} based on data dimensions")
            
            # Apply SVD
            svd = TruncatedSVD(n_components=actual_components, random_state=42)
            svd_result = svd.fit_transform(user_matrix)
            
            # Get the item factors (how each book relates to the latent factors)
            item_factors = svd.components_.T
            
            # Calculate variance explained
            variance_explained = svd.explained_variance_ratio_.sum()
            
            if verbose:
                print(f"  - SVD with {actual_components} components explaining {variance_explained:.2%} of variance")
                
            # Store the SVD model and user means for future predictions
            self.svd_model = svd
            self.user_means_map = user_means.to_dict()
            
            # Calculate item-item similarities
            if verbose:
                print("Step 4/4: Computing item similarity matrices")
                
            # Calculate item-item similarity using cosine similarity on the latent factors
            item_similarity = cosine_similarity(item_factors)
            
            # Create DataFrame for the similarity matrix
            self.item_similarity_matrix = pd.DataFrame(
                item_similarity,
                index=self.user_book_matrix.columns,
                columns=self.user_book_matrix.columns
            )
            
            if verbose:
                print(f"Collaborative filtering model built successfully")
                print(f"  - User-item matrix: {self.user_book_matrix.shape[0]} users x {self.user_book_matrix.shape[1]} books")
                print(f"  - Sparsity: {1.0 - np.count_nonzero(self.user_book_matrix.values) / np.prod(self.user_book_matrix.shape):.2%}")
                print(f"  - Item-item similarity matrix: {self.item_similarity_matrix.shape[0]} x {self.item_similarity_matrix.shape[1]}")
                
                # Sample book similarities
                sample_book = np.random.choice(self.item_similarity_matrix.index)
                similar_books = self.item_similarity_matrix[sample_book].sort_values(ascending=False)[1:6]
                print(f"\nSample collaborative filtering similar books to '{sample_book}':")
                for book, score in similar_books.items():
                    print(f"  - {book} (similarity: {score:.2f})")
                    
        except Exception as e:
            print(f"Error building collaborative filtering model: {str(e)}")
            # Create an identity matrix as fallback
            self.item_similarity_matrix = pd.DataFrame(
                np.eye(len(self.user_book_matrix.columns)),
                index=self.user_book_matrix.columns,
                columns=self.user_book_matrix.columns
            )
        
    def get_content_recommendations(self, book_title, n=10):
        """Get content-based recommendations for a book"""
        if book_title not in self.book_content_matrix.index:
            return pd.Series()
            
        sim_scores = self.book_content_matrix[book_title]
        sim_scores = sim_scores.sort_values(ascending=False)
        return sim_scores[1:n+1]  # Exclude the book itself
        
    def get_collaborative_recommendations(self, book_title, n=10):
        """Get collaborative filtering recommendations for a book"""
        if book_title not in self.item_similarity_matrix.index:
            return pd.Series()
            
        sim_scores = self.item_similarity_matrix[book_title]
        sim_scores = sim_scores.sort_values(ascending=False)
        return sim_scores[1:n+1]  # Exclude the book itself
        
    def get_popular_recommendations(self, n=10):
        """Get popularity-based recommendations"""
        # Ensure popular_books is properly sorted by score
        if hasattr(self, 'popular_books') and not self.popular_books.empty:
            return self.popular_books.head(n)
        elif hasattr(self, 'book_popularity_map') and 'count' in self.book_popularity_map and 'mean' in self.book_popularity_map:
            # Create a DataFrame from the book_popularity_map
            book_stats = pd.DataFrame({
                'count': self.book_popularity_map['count'],
                'mean': self.book_popularity_map['mean']
            })
            # Calculate score as count * mean
            book_stats['score'] = book_stats['count'] * book_stats['mean']
            # Sort by score
            return book_stats.sort_values('score', ascending=False).head(n)
        else:
            # Fallback if no popularity data available
            return pd.DataFrame(columns=['count', 'mean', 'score'])
        
    def get_hybrid_recommendations(self, book_title, n=10):
        """Get hybrid recommendations combining all approaches"""
        # Get recommendations from each model
        content_recs = self.get_content_recommendations(book_title, n=n*2)  # Get more to ensure diversity
        collab_recs = self.get_collaborative_recommendations(book_title, n=n*2)
        popular_recs = self.get_popular_recommendations(n=n*2)
        
        # If the book is not in our models, return popular recommendations
        if content_recs.empty and collab_recs.empty:
            if n > 0:
                print(f"Book '{book_title}' not found in dataset. Returning popular recommendations.")
            return popular_recs.index.tolist()[:n]
            
        # Normalize scores to 0-1 range for each model
        if not content_recs.empty:
            content_recs = (content_recs - content_recs.min()) / (content_recs.max() - content_recs.min()) if content_recs.max() != content_recs.min() else content_recs
            
        if not collab_recs.empty:
            collab_recs = (collab_recs - collab_recs.min()) / (collab_recs.max() - collab_recs.min()) if collab_recs.max() != collab_recs.min() else collab_recs
            
        # Get popularity scores and normalize
        pop_scores = pd.Series(0, index=popular_recs.index)
        if not popular_recs.empty and 'score' in popular_recs.columns:
            pop_scores = popular_recs['score']
            # Normalize popularity scores
            if pop_scores.max() != pop_scores.min():
                pop_scores = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min())
        
        # Collect all unique books from all recommendation sources
        all_books = set()
        if not content_recs.empty:
            all_books.update(content_recs.index)
        if not collab_recs.empty:
            all_books.update(collab_recs.index)
        all_books.update(popular_recs.index)
        
        # Remove the input book if it's in the recommendations
        if book_title in all_books:
            all_books.remove(book_title)
        
        # Combine the recommendations with weights
        hybrid_scores = {}
        
        for book in all_books:
            score = 0
            score_components = []
            
            # Add content-based score
            if book in content_recs and self.content_weight > 0:
                content_score = content_recs[book] * self.content_weight
                score += content_score
                score_components.append(f"C:{content_score:.2f}")
                
            # Add collaborative filtering score
            if book in collab_recs and self.collab_weight > 0:
                collab_score = collab_recs[book] * self.collab_weight
                score += collab_score
                score_components.append(f"L:{collab_score:.2f}")
                
            # Add popularity score
            if book in pop_scores.index and self.popular_weight > 0:
                pop_score = pop_scores[book] * self.popular_weight
                score += pop_score
                score_components.append(f"P:{pop_score:.2f}")
                
            hybrid_scores[book] = score
            
        # Sort and return top N recommendations
        sorted_books = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        # For debugging weight effectiveness, uncomment to see how scores are combined
        # if n > 0:
        #     print(f"\nTop recommendations for {book_title} with weights C:{self.content_weight:.2f} L:{self.collab_weight:.2f} P:{self.popular_weight:.2f}")
        #     for i, (book, score) in enumerate(sorted_books[:min(5, len(sorted_books))]):
        #         print(f"{i+1}. {book} (score: {score:.2f})")
                
        return [book for book, score in sorted_books[:n]]
        
    def predict_rating(self, user_id, book_title):
        """
        Predict a rating for a user-book pair using collaborative filtering, content-based, 
        and fallback approaches for cold-start cases.
        
        Args:
            user_id: User ID
            book_title: Book title
            
        Returns:
            Predicted rating on a 1-10 scale
        """
        # Check if we have data for this book and user
        user_data = self.merged_df[self.merged_df['User-ID'] == user_id]
        
        # Check for direct rating (user has already rated this book)
        direct_rating = user_data[user_data['Book-Title'] == book_title]
        if not direct_rating.empty:
            return direct_rating.iloc[0]['Book-Rating']
        
        # Cold-start handling for new users (less than threshold ratings)
        if len(user_data) < self.cold_start_threshold:
            # If new user, use book average or global average
            if book_title in self.popular_books.index:
                return self.popular_books.loc[book_title, 'mean']
            else:
                return self.global_mean_rating
        
        # Get similar books the user has rated
        user_ratings = user_data.sort_values('Book-Rating', ascending=False)
        
        if len(user_ratings) > 0:
            # APPROACH 1: Use collaborative filtering with similar books the user has rated
            rated_books = user_ratings[user_ratings['Book-Rating'] >= 7]['Book-Title'].tolist()
            
            if rated_books and book_title in self.item_similarity_matrix.index:
                # For each highly rated book, get its similarity to our target book
                similarities = []
                ratings = []
                
                for rated_book in rated_books[:5]:  # Use up to 5 books for efficiency
                    # Check in collaborative matrix
                    if rated_book in self.item_similarity_matrix.columns:
                        sim = self.item_similarity_matrix.loc[book_title, rated_book]
                        user_rating = user_ratings[user_ratings['Book-Title'] == rated_book].iloc[0]['Book-Rating']
                        
                        # Only use significant similarities
                        if sim > 0.1:  # Threshold for meaningful similarity
                            similarities.append(sim)
                            ratings.append(user_rating)
                
                # If we have similarities, compute weighted average
                if similarities and len(similarities) >= 2:
                    similarities = np.array(similarities)
                    ratings = np.array(ratings)
                    # Weighted average based on similarity
                    weighted_rating = np.sum(similarities * ratings) / np.sum(similarities) if np.sum(similarities) > 0 else 0
                    
                    # Add user's mean rating bias if available
                    if hasattr(self, 'user_means_map') and user_id in self.user_means_map:
                        # Adjust prediction with user's rating tendency
                        baseline = self.user_means_map[user_id]
                        # Mix weighted prediction with baseline
                        final_rating = 0.8 * weighted_rating + 0.2 * baseline
                    else:
                        final_rating = weighted_rating
                    
                    # Ensure rating is within explicit rating range (1-10)
                    return max(1, min(10, final_rating))
            
            # APPROACH 2: Use hybrid recommendations position as a proxy for rating
            seed_book = user_ratings.iloc[0]['Book-Title']
            recommendations = self.get_hybrid_recommendations(seed_book, n=50)
            
            # If the book is in recommendations, estimate rating based on position
            if book_title in recommendations:
                position = recommendations.index(book_title)
                position_ratio = position / len(recommendations)
                
                # Explicit rating range is 1-10, so map position to this range
                # Higher positions (lower index) get higher ratings
                rating = 10 - position_ratio * 9  # Map from 10 down to 1
                
                # Blend with user's average rating tendency
                if hasattr(self, 'user_means_map') and user_id in self.user_means_map:
                    user_mean = self.user_means_map[user_id]
                    # Mix position-based rating with user's average
                    blended_rating = 0.7 * rating + 0.3 * user_mean
                    return max(1, min(10, blended_rating))
                
                return max(1, min(10, rating))
            
        # APPROACH 3: Fall back to baseline approaches
        # Try to use a mix of user and book averages
        user_mean = None
        book_mean = None
        
        # Get user's mean rating if available
        if hasattr(self, 'user_means_map') and user_id in self.user_means_map:
            user_mean = self.user_means_map[user_id]
        elif user_id in self.user_mean_ratings:
            user_mean = self.user_mean_ratings[user_id]
            
        # Get book's mean rating if available
        if book_title in self.popular_books.index:
            book_mean = self.popular_books.loc[book_title, 'mean']
            
        # Combine available information with appropriate weights
        if user_mean is not None and book_mean is not None:
            # Blend user and book means
            return 0.6 * user_mean + 0.4 * book_mean
        elif book_mean is not None:
            return book_mean
        elif user_mean is not None:
            return user_mean
        else:
            # Ultimate fallback is global average
            return self.global_mean_rating
        
    def split_data(self, test_size=0.2, random_state=42, verbose=True):
        """
        Split data into training and test sets for evaluation
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed information
        
        Returns:
            Tuple of (train_data, test_data)
        """
        if verbose:
            print("Splitting data into training and test sets...")
            
        # Get all explicit ratings (rating > 0)
        ratings_data = self.merged_df[['User-ID', 'Book-Title', 'Book-Rating']]
        ratings_data = ratings_data[ratings_data['Book-Rating'] > 0]
        
        # Split data
        self.train_data, self.test_data = train_test_split(
            ratings_data, test_size=test_size, random_state=random_state
        )
        
        if verbose:
            print(f"  - Training set: {len(self.train_data)} ratings")
            print(f"  - Test set: {len(self.test_data)} ratings")
        
        # Compute global statistics from training data only
        self.global_mean_rating = self.train_data['Book-Rating'].mean()
        self.rating_std = self.train_data['Book-Rating'].std()
        
        if verbose:
            print(f"  - Global rating statistics: mean={self.global_mean_rating:.2f}, std={self.rating_std:.2f}")
            print(f"  - Rating range: {self.train_data['Book-Rating'].min()}-{self.train_data['Book-Rating'].max()}")
        
        # Compute user mean ratings (for users with sufficient ratings)
        user_stats = self.train_data.groupby('User-ID')['Book-Rating'].agg(['count', 'mean', 'std'])
        
        # Only store means for users with at least 3 ratings (more reliable)
        reliable_users = user_stats[user_stats['count'] >= 3]
        self.user_mean_ratings = reliable_users['mean'].to_dict()
        
        if verbose:
            print(f"  - User statistics: {len(user_stats)} users, {len(reliable_users)} with reliable means")
            print(f"  - User mean ratings range: {reliable_users['mean'].min():.2f}-{reliable_users['mean'].max():.2f}")
        
        # Compute book statistics from training data
        book_stats = self.train_data.groupby('Book-Title')['Book-Rating'].agg(['count', 'mean', 'std'])
        
        # Only use books with at least 3 ratings for reliable statistics
        reliable_books = book_stats[book_stats['count'] >= 3]
        self.book_popularity_map = {
            'count': reliable_books['count'].to_dict(),
            'mean': reliable_books['mean'].to_dict(),
            'std': reliable_books['std'].to_dict()
        }
        
        if verbose:
            print(f"  - Book statistics: {len(book_stats)} books, {len(reliable_books)} with reliable means")
            if len(reliable_books) > 0:
                print(f"  - Book mean ratings range: {reliable_books['mean'].min():.2f}-{reliable_books['mean'].max():.2f}")
        
        # Create mapping of users to their rated books (for fast lookups)
        self.user_books_map = {}
        for user_id, group in self.train_data.groupby('User-ID'):
            self.user_books_map[user_id] = set(group['Book-Title'].tolist())
            
        # Create mapping of books to users who rated them (for fast lookups)
        self.book_users_map = {}
        for book_title, group in self.train_data.groupby('Book-Title'):
            self.book_users_map[book_title] = set(group['User-ID'].tolist())
            
        return self.train_data, self.test_data
        
    def evaluate_model(self, k=10, verbose=True):
        """Evaluate the model using various metrics"""
        if self.test_data is None:
            print("Error: No test data available. Run split_data() first.")
            return
            
        # Prediction metrics
        test_users = self.test_data['User-ID'].unique()
        # Limit to a reasonable number of users for faster evaluation
        max_test_users = min(100, len(test_users))
        test_users = np.random.choice(test_users, max_test_users, replace=False)
        
        test_predictions = []
        test_actual = []
        
        # Ranking metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        # Save current weights to restore later
        original_weights = (self.content_weight, self.collab_weight, self.popular_weight)
        weights_str = f"C:{self.content_weight:.2f}_L:{self.collab_weight:.2f}_P:{self.popular_weight:.2f}"
        
        for user_idx, user_id in enumerate(test_users):
            if verbose and user_idx % 10 == 0:
                print(f"Evaluating user {user_idx+1}/{len(test_users)}...")
                
            # Get actual ratings for this user
            user_test_data = self.test_data[self.test_data['User-ID'] == user_id]
            
            # For prediction metrics
            for _, row in user_test_data.iterrows():
                try:
                    predicted = self.predict_rating(user_id, row['Book-Title'])
                    test_predictions.append(predicted)
                    test_actual.append(row['Book-Rating'])
                except Exception as e:
                    # Skip problematic predictions but log the error
                    # print(f"Error predicting rating for user {user_id}, book {row['Book-Title']}: {str(e)}")
                    continue
            
            # For ranking metrics
            # Get top rated books for this user as relevant items (books rated 7+)
            relevant_books = user_test_data[user_test_data['Book-Rating'] >= 7]['Book-Title'].tolist()
            
            if relevant_books:
                # Get user's highly rated books from training data as seed for recommendations
                user_train_data = self.train_data[self.train_data['User-ID'] == user_id]
                if len(user_train_data) > 0:
                    try:
                        # Sort by rating to get the highest rated book
                        seed_book = user_train_data.sort_values('Book-Rating', ascending=False).iloc[0]['Book-Title']
                        
                        # Get recommendations
                        recommendations = self.get_hybrid_recommendations(seed_book, n=k)
                        
                        # Check if we have valid recommendations
                        if recommendations and len(recommendations) > 0:
                            # Calculate ranking metrics
                            precision = self.precision_at_k(recommendations, relevant_books, k)
                            recall = self.recall_at_k(recommendations, relevant_books, k)
                            f1 = self.f1_score_at_k(recommendations, relevant_books, k)
                            
                            precision_scores.append(precision)
                            recall_scores.append(recall)
                            f1_scores.append(f1)
                    except Exception as e:
                        # Skip problematic recommendations but log the error
                        # print(f"Error generating recommendations for user {user_id}: {str(e)}")
                        continue
        
        # Calculate metrics
        if len(test_predictions) == 0 or len(precision_scores) == 0:
            print(f"Warning: Not enough data for evaluation with weights {weights_str}")
            # Return poor metrics to discourage these weights
            return {
                'MAE': 10.0,
                'RMSE': 10.0,
                'Precision@K': 0.0,
                'Recall@K': 0.0,
                'F1@K': 0.0
            }
        
        # Calculate and return metrics
        evaluation = {
            'MAE': self.mae(test_predictions, test_actual)*1000000,
            'RMSE': self.rmse(test_predictions, test_actual),
            'Precision@K': np.mean(precision_scores),
            'Recall@K': np.mean(recall_scores),
            'F1@K': np.mean(f1_scores),
            'NumPredictions': len(test_predictions),
            'NumRankingEvals': len(precision_scores)
        }
        
        if verbose:
            print(f"\nModel Evaluation with weights {weights_str}:")
            for metric, value in evaluation.items():
                if metric not in ['NumPredictions', 'NumRankingEvals']:
                    print(f"{metric}: {value:.4f}")
            print(f"Based on {evaluation['NumPredictions']} predictions and {evaluation['NumRankingEvals']} ranking evaluations")
            
        print(f"Collected {len(test_predictions)} predictions for RMSE calculation")
        
        return evaluation
        
    def optimize_model_weights(self):
        """Optimize the weights for different recommendation components"""
        if self.test_data is None or not self.optimize_weights:
            return
            
        print("Optimizing model weights...")
        
        # Keep track of the best weights and performance
        best_weights = [self.content_weight, self.collab_weight, self.popular_weight]
        best_performance = float('inf')  # Lower is better for our optimization objective
        
        # Define a more effective objective function that ensures weights actually change
        def objective_function(weights):
            # Normalize weights to sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Store original weights
            original_weights = [self.content_weight, self.collab_weight, self.popular_weight]
            
            # Set new weights
            self.content_weight = weights[0]
            self.collab_weight = weights[1]
            self.popular_weight = weights[2]
            
            # Skip evaluation if weights are almost identical to original
            if np.allclose(weights, original_weights, rtol=1e-3, atol=1e-3):
                # Add a penalty to encourage exploration of different weights
                return 10.0  # High penalty value
            
            # Evaluate with these weights
            eval_results = self.evaluate_model(verbose=False)
            
            # Objective: optimize a weighted combination of metrics
            # We want to minimize RMSE and maximize F1@K
            # Lower objective value is better
            objective_value = eval_results['RMSE'] * 0.5 - eval_results['F1@K'] * 0.5
                
            return objective_value

        # Try multiple optimization runs with different starting points
        bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]  # Ensure no weight is 0 or 1
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Try different starting points
        starting_points = [
            [0.33, 0.33, 0.34],  # Equal weights
            [0.6, 0.3, 0.1],     # Content-heavy
            [0.2, 0.7, 0.1],     # Collaborative-heavy
            [0.1, 0.1, 0.8],     # Popularity-heavy
            [self.content_weight, self.collab_weight, self.popular_weight]  # Current weights
        ]
        
        for start_point in starting_points:
            result = minimize(
                objective_function, 
                start_point,
                bounds=bounds,
                constraints=constraint,
                method='SLSQP',
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            
            # Check if this result is better than our current best
            if result.success and result.fun < best_performance:
                best_performance = result.fun
                best_weights = result.x / np.sum(result.x)  # Normalize
        
        # Apply the best weights found
        self.content_weight = best_weights[0]
        self.collab_weight = best_weights[1]
        self.popular_weight = best_weights[2]
        
        print(f"Optimized weights: Content={self.content_weight:.2f}, Collaborative={self.collab_weight:.2f}, Popularity={self.popular_weight:.2f}")
        
    def get_book_details(self, book_titles):
        """Get details for a list of book titles"""
        books_info = []
        for title in book_titles:
            book_data = self.merged_df[self.merged_df['Book-Title'] == title].iloc[0]
            books_info.append({
                'Title': book_data['Book-Title'],
                'Author': book_data['Book-Author'],
                'Year': book_data['Year-Of-Publication'],
                'ISBN': book_data['ISBN'],
                'Publisher': book_data['Publisher'],
                'Cover': book_data.get('Image-URL-L', None)
            })
        return books_info
        
    def save_model(self, filepath='book_recommender_model.pkl'):
        """Save the trained model"""
        model_data = {
            'content_weight': self.content_weight,
            'collab_weight': self.collab_weight,
            'popular_weight': self.popular_weight,
            'optimize_weights': self.optimize_weights,
            'item_similarity_matrix': self.item_similarity_matrix,
            'book_content_matrix': self.book_content_matrix,
            'popular_books': self.popular_books,
            'merged_df': self.merged_df[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]
                .drop_duplicates('Book-Title'),
            'cold_start_threshold': self.cold_start_threshold,
            'global_mean_rating': self.global_mean_rating,
            'user_mean_ratings': self.user_mean_ratings,
            'book_popularity_map': self.book_popularity_map
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath='book_recommender_model.pkl'):
        """Load a previously trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.content_weight = model_data['content_weight']
        self.collab_weight = model_data['collab_weight']
        self.popular_weight = model_data['popular_weight']
        self.optimize_weights = model_data.get('optimize_weights', False)
        self.item_similarity_matrix = model_data['item_similarity_matrix']
        self.book_content_matrix = model_data['book_content_matrix']
        self.popular_books = model_data['popular_books']
        self.merged_df = model_data['merged_df']
        self.cold_start_threshold = model_data.get('cold_start_threshold', 5)
        self.global_mean_rating = model_data.get('global_mean_rating', 0)
        self.user_mean_ratings = model_data.get('user_mean_ratings', {})
        self.book_popularity_map = model_data.get('book_popularity_map', {})
        
        print(f"Model loaded from {filepath}")

    # Add evaluation metrics
    def mae(self, predictions, true_ratings):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(np.array(predictions) - np.array(true_ratings)))
        
    def rmse(self, predictions, true_ratings):
        """Calculate Root Mean Square Error"""
        return np.sqrt(np.mean(np.square(np.array(predictions) - np.array(true_ratings))))
        
    def precision_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate precision at k"""
        if len(recommended_items) == 0:
            return 0
        recommended_k = recommended_items[:k]
        relevant_and_recommended = set(recommended_k) & set(relevant_items)
        return len(relevant_and_recommended) / min(k, len(recommended_k))
        
    def recall_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate recall at k"""
        if len(relevant_items) == 0:
            return 0
        recommended_k = recommended_items[:k]
        relevant_and_recommended = set(recommended_k) & set(relevant_items)
        return len(relevant_and_recommended) / len(relevant_items)
        
    def f1_score_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate F1 score at k"""
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

# Usage example
if __name__ == "__main__":
    try:
        print("\n===== INITIALIZING RECOMMENDER SYSTEM =====")
        recommender = HybridBookRecommender(
            content_weight=0.3,
            collab_weight=0.5,
            popular_weight=0.2,
            optimize_weights=True
        )
        
        print("\n===== LOADING DATASETS =====")
        # Load data
        recommender.load_data(
            books_path='Recommender/Books.csv',
            ratings_path='Recommender/Ratings.csv',
            users_path='Recommender/Users.csv'
        )
        
        print("\n===== PREPROCESSING DATA =====")
        # Detailed preprocessing with verbose output
        processed_data = recommender.preprocess_data(
            min_book_ratings=5,   # Include books with at least 15 ratings
            min_user_ratings=3,    # Include users with at least 3 ratings
            verbose=True           # Show detailed preprocessing steps
        )
        
        print("\n===== SPLITTING DATA FOR EVALUATION =====")
        train_data, test_data = recommender.split_data(test_size=0.2, random_state=42, verbose=True)
        print(f"Training data: {len(train_data)} ratings | Test data: {len(test_data)} ratings")
        
        print("\n===== BUILDING CONTENT-BASED MODEL =====")
        # Build content-based model with enhanced features
        recommender.build_content_based_model(verbose=True)
        
        print("\n===== BUILDING COLLABORATIVE FILTERING MODEL =====")
        # Build collaborative filtering model with improved techniques
        recommender.build_collaborative_model(n_components=50, verbose=True)
        
        print("\n===== EVALUATING DIFFERENT WEIGHT CONFIGURATIONS =====")
        # Try different weight configurations to see their impact
        weight_configs = [
            {"content": 0.6, "collab": 0.3, "popular": 0.1, "name": "Content-heavy"},
            {"content": 0.3, "collab": 0.6, "popular": 0.1, "name": "Collaborative-heavy"},
            {"content": 0.2, "collab": 0.2, "popular": 0.6, "name": "Popularity-heavy"},
            {"content": 0.33, "collab": 0.33, "popular": 0.34, "name": "Balanced"}
        ]
        
        results = {}
        print("\nTesting different weight configurations:")
        for config in weight_configs:
            # Set weights
            recommender.content_weight = config["content"]
            recommender.collab_weight = config["collab"]
            recommender.popular_weight = config["popular"]
            
            # Evaluate with limited verbosity
            print(f"\nEvaluating {config['name']} weights (C:{config['content']:.2f}, L:{config['collab']:.2f}, P:{config['popular']:.2f}):")
            results[config["name"]] = recommender.evaluate_model(k=10, verbose=False)
            # Print key metrics
            for metric in ["RMSE","MAE", "Precision@K", "Recall@K", "F1@K"]:
                print(f"{metric}: {results[config['name']][metric]:.4f}")
        
        # Find best configuration for F1 score
        best_f1_config = max(results.items(), key=lambda x: x[1]["F1@K"])
        print(f"\nBest configuration for F1@K: {best_f1_config[0]} with F1@K={best_f1_config[1]['F1@K']:.4f}")
        
        print("\n===== OPTIMIZING WEIGHTS =====")
        print("Running weight optimization...")
        # Set starting weights to the best manual configuration
        for config in weight_configs:
            if config["name"] == best_f1_config[0]:
                recommender.content_weight = config["content"]
                recommender.collab_weight = config["collab"]
                recommender.popular_weight = config["popular"]
                break
                
        # Run optimization
        recommender.optimize_model_weights()
        
        # Evaluate with optimized weights
        print("\nEvaluating with optimized weights:")
        optimized_results = recommender.evaluate_model(k=10, verbose=True)
        
        # Compare with best manual configuration
        print("\nComparison with best manual configuration:")
        for metric in ["RMSE", "Precision@K", "Recall@K", "F1@K"]:
            manual = best_f1_config[1][metric]
            optimized = optimized_results[metric]
            diff = optimized - manual
            better = (metric != "RMSE" and diff > 0) or (metric == "RMSE" and diff < 0)
            print(f"{metric}: {manual:.4f} → {optimized:.4f} ({'better' if better else 'worse'} by {abs(diff):.4f})")
        
        print("\n===== RECOMMENDATION EXAMPLES =====")
        # Example book titles to demonstrate
        example_books = ["The Da Vinci Code", "Harry Potter and the Sorcerer's Stone", "To Kill a Mockingbird"]
        for book in example_books:
            if book in recommender.merged_df['Book-Title'].values:
                print(f"\nRecommendations for '{book}':")
                
                # Get recommendations from each component model
                content_recs = recommender.get_content_recommendations(book, n=3)
                collab_recs = recommender.get_collaborative_recommendations(book, n=3)
                popular_recs = recommender.get_popular_recommendations(n=3)
                
                print("\nContent-based recommendations:")
                for i, (rec_book, score) in enumerate(content_recs.items(), 1):
                    print(f"{i}. {rec_book} (similarity: {score:.2f})")
                    
                print("\nCollaborative filtering recommendations:")
                for i, (rec_book, score) in enumerate(collab_recs.items(), 1):
                    print(f"{i}. {rec_book} (similarity: {score:.2f})")
                    
                print("\nPopularity-based recommendations:")
                for i, book_title in enumerate(popular_recs.index[:3], 1):
                    if 'count' in popular_recs.columns and 'mean' in popular_recs.columns:
                        count = popular_recs.loc[book_title, 'count']
                        mean = popular_recs.loc[book_title, 'mean']
                        print(f"{i}. {book_title} (count: {count}, avg: {mean:.2f})")
                    else:
                        print(f"{i}. {book_title}")
                
                # Get hybrid recommendations
                print("\nHybrid recommendations:")
                hybrid_recs = recommender.get_hybrid_recommendations(book, n=5)
                for i, rec in enumerate(hybrid_recs, 1):
                    print(f"{i}. {rec}")
            else:
                print(f"\n'{book}' not found in the dataset.")
        
        # Demonstrate cold-start handling
        print("\n===== COLD-START HANDLING DEMONSTRATION =====")
        # Find a user with few ratings
        cold_start_user = min(recommender.user_mean_ratings.keys(), key=lambda x: len(recommender.merged_df[recommender.merged_df['User-ID'] == x]))
        print(f"For cold-start user (User ID {cold_start_user}):")
        # Get a book the user hasn't rated
        user_books = recommender.merged_df[recommender.merged_df['User-ID'] == cold_start_user]['Book-Title'].tolist()
        
        # Try to find an unrated popular book
        try:
            # Get top popular books
            top_popular_books = recommender.get_popular_recommendations(n=20).index.tolist()
            # Find first book user hasn't rated
            unrated_book = next(book for book in top_popular_books if book not in user_books)
            
            predicted_rating = recommender.predict_rating(cold_start_user, unrated_book)
            print(f"Predicted rating for '{unrated_book}': {predicted_rating:.2f}")
            
            # Get user's actual rated books and their ratings
            print("\nUser's actual ratings:")
            user_ratings = recommender.merged_df[recommender.merged_df['User-ID'] == cold_start_user][['Book-Title', 'Book-Rating']].sort_values('Book-Rating', ascending=False)
            for i, (_, row) in enumerate(user_ratings.head(3).iterrows(), 1):
                print(f"{i}. '{row['Book-Title']}': {row['Book-Rating']}")
            
            # Show recommendations for this cold-start user
            print("\nRecommendations for cold-start user:")
            if user_books:
                seed_book = user_ratings.iloc[0]['Book-Title']
                cold_start_recs = recommender.get_hybrid_recommendations(seed_book, n=5)
                for i, rec in enumerate(cold_start_recs, 1):
                    print(f"{i}. {rec}")
        except (StopIteration, IndexError):
            print("Could not find an unrated popular book for demonstration.")
        
        print("\n===== SAVING MODEL =====")
        # Save model for future use
        recommender.save_model('Recommender/book_recommender_model.pkl')
        
        print("\nRecommender system setup and evaluation complete!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc() 