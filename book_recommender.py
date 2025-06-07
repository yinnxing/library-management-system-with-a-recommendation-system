import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import os

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class BookRecommender:
    def __init__(self, books_path, ratings_path, users_path):
        """Initialize the recommender system with dataset paths."""
        self.books_path = books_path
        self.ratings_path = ratings_path
        self.users_path = users_path
        self.books_df = None
        self.ratings_df = None
        self.users_df = None
        self.cf_algo = None
        self.content_model = None
        self.tfidf_matrix = None
        self.hybrid_weight = 0.6  # Weight for collaborative filtering (0.4 for content-based)

    def load_data(self):
        """Load datasets from CSV files."""
        print("Loading datasets...")
        self.books_df = pd.read_csv(self.books_path, sep=',', on_bad_lines='skip', encoding='latin-1')
        self.ratings_df = pd.read_csv(self.ratings_path, sep=',', on_bad_lines='skip')
        self.users_df = pd.read_csv(self.users_path, sep=',', on_bad_lines='skip')
        
        # Clean and preprocess data
        self._preprocess_data()
        
        print(f"Loaded {len(self.books_df)} books, {len(self.ratings_df)} ratings, and {len(self.users_df)} users.")
        
    def _preprocess_data(self):
        """Preprocess and clean the data."""
        # Remove rows with missing values in essential columns
        self.books_df = self.books_df.dropna(subset=['ISBN', 'Book-Title', 'Book-Author'])
        
        # Convert ratings to explicit feedback (ratings 0 are implicit and will be removed)
        self.ratings_df = self.ratings_df[self.ratings_df['Book-Rating'] > 0]
        
        # Filter out books with less than 10 reviews
        book_review_counts = self.ratings_df['ISBN'].value_counts()
        popular_books = book_review_counts[book_review_counts >= 10].index
        self.ratings_df = self.ratings_df[self.ratings_df['ISBN'].isin(popular_books)]
        
        # Filter out users with less than 3 reviews
        user_review_counts = self.ratings_df['User-ID'].value_counts()
        active_users = user_review_counts[user_review_counts >= 3].index
        self.ratings_df = self.ratings_df[self.ratings_df['User-ID'].isin(active_users)]
        
        # Ensure all ISBNs in ratings exist in books dataset
        self.ratings_df = self.ratings_df[self.ratings_df['ISBN'].isin(self.books_df['ISBN'])]
        
        # Print filtering statistics
        books_removed = len(self.books_df) - len(self.books_df[self.books_df['ISBN'].isin(self.ratings_df['ISBN'])])
        users_removed = len(self.users_df) - len(self.users_df[self.users_df['User-ID'].isin(self.ratings_df['User-ID'])])
        
        print(f"Filtered out {books_removed} books with less than 10 reviews")
        print(f"Filtered out {users_removed} users with less than 3 reviews")
        print(f"Remaining ratings: {len(self.ratings_df)}")
        
        # Prepare metadata for content-based filtering
        self.books_df['content'] = self.books_df['Book-Title'] + ' ' + self.books_df['Book-Author'] + ' ' + self.books_df['Publisher'].fillna('')
        
    def train_content_based_model(self, max_books=5000):
        """Train the content-based recommendation model using TF-IDF.
        
        Args:
            max_books (int): Maximum number of books to include in the content model to avoid memory issues.
        """
        print("Training content-based model...")
        
        # If we have too many books, limit to the most popular ones to avoid memory issues
        if len(self.books_df) > max_books:
            # Get book popularity
            popular_books = self.ratings_df['ISBN'].value_counts().head(max_books).index
            print(f"Limiting content-based model to {max_books} most popular books to save memory.")
            
            # Only use popular books for content model
            content_books_df = self.books_df[self.books_df['ISBN'].isin(popular_books)].copy()
        else:
            content_books_df = self.books_df.copy()
        
        # Verify we have books to process
        if len(content_books_df) == 0:
            print("Warning: No books available for content-based modeling. Skipping content-based model training.")
            self.content_book_indices = {}
            return
            
        # Create a mapping from ISBN to index position in the content matrix
        # This will help us track which books are included in the content model
        isbn_to_content_idx = {isbn: idx for idx, isbn in enumerate(content_books_df['ISBN'])}
        
        # Use TF-IDF for text features with reduced parameters
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,  # Limit features
            min_df=2,           # Ignore terms that appear in fewer than 2 documents
            max_df=0.85         # Ignore terms that appear in more than 85% of documents
        )
        
        # Process in smaller chunks if necessary
        chunk_size = 1000
        if len(content_books_df) > chunk_size:
            print(f"Processing TF-IDF in chunks of {chunk_size} books...")
            
            # Process books in chunks to avoid memory issues
            chunks = [content_books_df['content'].iloc[i:i+chunk_size] for i in range(0, len(content_books_df), chunk_size)]
            
            # Fit on the first chunk
            self.tfidf_matrix = tfidf.fit_transform(chunks[0])
            
            # Transform remaining chunks and stack vertically
            for i, chunk in enumerate(chunks[1:], 1):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                chunk_matrix = tfidf.transform(chunk)
                from scipy.sparse import vstack
                self.tfidf_matrix = vstack([self.tfidf_matrix, chunk_matrix])
        else:
            # Process all at once if it's a small dataset
            self.tfidf_matrix = tfidf.fit_transform(content_books_df['content'])
        
        # Create a mapping from original books_df indices to content matrix indices
        self.content_book_indices = {}
        
        # Iterate through all books in the books_df
        for idx, row in self.books_df.iterrows():
            isbn = row['ISBN']
            # Check if this ISBN is in our content model
            if isbn in isbn_to_content_idx:
                self.content_book_indices[idx] = isbn_to_content_idx[isbn]
        
        print(f"Created mapping for {len(self.content_book_indices)} books between original index and content model index.")
        
        # Compute similarity matrix with reduced precision to save memory
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        print("Computing similarity matrix...")
        self.content_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix, dense_output=False)
        
        # Convert to float32 to reduce memory usage
        if not isinstance(self.content_sim_matrix, np.ndarray):
            self.content_sim_matrix = self.content_sim_matrix.toarray().astype(np.float32)
        else:
            self.content_sim_matrix = self.content_sim_matrix.astype(np.float32)
            
        print(f"Content-based model trained successfully with {self.tfidf_matrix.shape[0]} books.")
        
    def train_collaborative_filtering(self, algorithm='svd'):
        """Train the collaborative filtering model."""
        print("Training collaborative filtering model...")
        # Create a Surprise dataset
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(self.ratings_df[['User-ID', 'ISBN', 'Book-Rating']], reader)
        
        # Split data for training
        self.trainset, self.testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
        
        # Choose algorithm with improved parameters
        if algorithm == 'svd':
            # Increase factors, epochs and adjust regularization for better performance
            self.cf_algo = SVD(
                n_factors=150,         # Increased from 100
                n_epochs=50,           # Increased from 20
                lr_all=0.008,          # Increased from 0.005
                reg_all=0.01,          # Reduced from 0.02
                init_mean=0,
                init_std_dev=0.05,
                random_state=42
            )
        elif algorithm == 'knn':
            # Use item-based KNN which often performs better for book recommendations
            self.cf_algo = KNNBasic(
                k=60,                  # Increased from 40
                min_k=3,               # Ensure at least 3 neighbors
                sim_options={
                    'name': 'pearson_baseline',  # Better than pearson
                    'user_based': False,         # Use item-based instead of user-based
                    'min_support': 3             # Minimum number of common items
                }
            )
        
        # Train the model
        print("Fitting collaborative filtering model...")
        self.cf_algo.fit(self.trainset)
        print("Collaborative filtering model trained successfully.")
        
    def evaluate_collaborative_filtering(self):
        """Evaluate the collaborative filtering model."""
        # Test the model
        predictions = self.cf_algo.test(self.testset)
        
        # Compute RMSE and MAE
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        print(f"Collaborative Filtering Evaluation: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        return rmse, mae
    
    def evaluate_metrics_at_k(self, k_values=[5, 10, 20]):
        """Evaluate Precision@K, Recall@K, and F1@K for different k values."""
        # Get unique users in the test set
        test_users = set()
        for uid, _, _, _, _ in self.cf_algo.test(self.testset):
            test_users.add(uid)
        
        # Limit to a reasonable number of users for evaluation
        max_test_users = min(100, len(test_users))
        if len(test_users) > max_test_users:
            import random
            random.seed(42)
            test_users = set(random.sample(list(test_users), max_test_users))
        
        print(f"Evaluating metrics on {len(test_users)} users")
        
        # For each user, get top-K recommendations
        results = {}
        for k in k_values:
            precisions = []
            recalls = []
            f1_scores = []
            
            for uid in test_users:
                # Get ground truth: items the user rated highly in the test set
                user_test_ratings = [(iid, true_r) for (u, iid, true_r, _, _) in self.cf_algo.test(self.testset) if u == uid]
                user_relevant_items = {iid for (iid, true_r) in user_test_ratings if true_r >= 8}
                
                if len(user_relevant_items) == 0:
                    continue
                
                # Generate realistic recommendations from the entire catalog
                # (similar to get_collaborative_recommendations but for evaluation)
                try:
                    # Get all items in the training set
                    items_in_trainset = self.trainset.all_items()
                    
                    # Convert inner ids to raw ids
                    all_items = [self.trainset.to_raw_iid(inner_id) for inner_id in items_in_trainset]
                    
                    # Get items the user has already rated in the training set
                    # This is more realistic as we shouldn't recommend items the user has already rated
                    user_rated_in_train = set()
                    if self.trainset.knows_user(self.trainset.to_inner_uid(uid)):
                        user_rated_in_train = set(
                            self.trainset.to_raw_iid(iid) 
                            for (iid, _) in self.trainset.ur[self.trainset.to_inner_uid(uid)]
                        )
                    
                    # Filter out items the user has already rated
                    items_to_predict = list(set(all_items) - user_rated_in_train)
                    
                    # Predict ratings for all unrated items
                    predictions = [self.cf_algo.predict(uid, item_id) for item_id in items_to_predict]
                    
                    # Sort by estimated rating
                    predictions.sort(key=lambda x: x.est, reverse=True)
                    
                    # Get top-K recommendations
                    top_k_recs = [pred.iid for pred in predictions[:k]]
                    
                    # Calculate precision and recall
                    n_rel_and_rec = len(set(top_k_recs) & user_relevant_items)
                    precision = n_rel_and_rec / k if k != 0 else 0
                    recall = n_rel_and_rec / len(user_relevant_items) if len(user_relevant_items) != 0 else 0
                    
                    # Calculate F1 score
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
                
                except Exception as e:
                    print(f"Error evaluating user {uid}: {str(e)}")
                    continue
            
            # Average metrics across users
            avg_precision = np.mean(precisions) if precisions else 0
            avg_recall = np.mean(recalls) if recalls else 0
            avg_f1 = np.mean(f1_scores) if f1_scores else 0
            
            print(f"At k={k}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")
            print(f"Based on {len(precisions)} users with relevant items in test set")
            results[k] = {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}
        
        # Plot metrics
        self._plot_metrics(results)
        
        return results
    
    def _plot_metrics(self, results):
        """Plot evaluation metrics for different k values."""
        k_values = list(results.keys())
        precision_values = [results[k]['precision'] for k in k_values]
        recall_values = [results[k]['recall'] for k in k_values]
        f1_values = [results[k]['f1'] for k in k_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, precision_values, 'o-', label='Precision@K')
        plt.plot(k_values, recall_values, 's-', label='Recall@K')
        plt.plot(k_values, f1_values, '^-', label='F1@K')
        plt.xlabel('K Values')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics at Different K Values')
        plt.legend()
        plt.grid(True)
        plt.savefig('evaluation_metrics.png')
        plt.close()
    
    def get_content_based_recommendations(self, book_id, n=10):
        """Get content-based recommendations for a specific book."""
        # Check if we have content indices
        if not hasattr(self, 'content_book_indices') or len(self.content_book_indices) == 0:
            print("No content model available. Cannot provide content-based recommendations.")
            # Return empty DataFrame with the same structure as books_df
            return self.books_df.head(0)
            
        if not isinstance(book_id, int):
            # If ISBN is provided instead of index
            matching_books = self.books_df[self.books_df['ISBN'] == book_id]
            if len(matching_books) == 0:
                print(f"Book with ISBN {book_id} not found in dataset.")
                return self.books_df.head(0)
            book_index = matching_books.index[0]
        else:
            book_index = book_id
        
        # Check if this book is in our content model
        if book_index not in self.content_book_indices:
            print(f"Book index {book_index} not in content model. Finding a proxy book.")
            
            # Try to get the book's details to find similar books
            try:
                book_details = self.books_df.loc[book_index]
                book_author = book_details['Book-Author']
                
                # Try to find books by the same author
                author_books = self.books_df[
                    (self.books_df['Book-Author'] == book_author) & 
                    (self.books_df.index.isin(self.content_book_indices.keys()))
                ]
                
                if len(author_books) > 0:
                    book_index = author_books.index[0]
                    print(f"Using another book by {book_author} as proxy.")
                else:
                    # If no books by same author, use the first available content book
                    book_index = list(self.content_book_indices.keys())[0]
                    proxy_details = self.books_df.loc[book_index]
                    print(f"Using random book as proxy: {proxy_details['Book-Title']} by {proxy_details['Book-Author']}")
            except:
                # If any error occurs, use the first available content book
                book_index = list(self.content_book_indices.keys())[0]
                print(f"Using first available book in content model as proxy.")
        
        # Get the corresponding index in the content matrix
        try:
            content_index = self.content_book_indices[book_index]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_sim_matrix[content_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n+1]  # Skip the book itself
            
            # Get indices in the content matrix
            content_indices = [i[0] for i in sim_scores]
            
            # Map back to original book indices
            reverse_mapping = {v: k for k, v in self.content_book_indices.items()}
            book_indices = [reverse_mapping[idx] for idx in content_indices if idx in reverse_mapping]
            
            if len(book_indices) == 0:
                print("No recommendations could be generated from the content model.")
                return self.books_df.head(0)
                
            return self.books_df.iloc[book_indices]
            
        except Exception as e:
            print(f"Error generating content-based recommendations: {e}")
            return self.books_df.head(0)
    
    def get_collaborative_recommendations(self, user_id, n=10):
        """Get collaborative filtering recommendations for a specific user."""
        # Get all items in the training set
        items_in_trainset = self.trainset.all_items()
        
        # Convert inner ids to raw ids
        items = [self.trainset.to_raw_iid(inner_id) for inner_id in items_in_trainset]
        
        # Get rated items by user
        user_rated_items = set([rating[1] for rating in self.ratings_df[self.ratings_df['User-ID'] == user_id].values])
        
        # Filter out items the user has already rated
        items_to_predict = list(set(items) - user_rated_items)
        
        # Predict ratings
        predictions = [self.cf_algo.predict(user_id, item_id) for item_id in items_to_predict]
        
        # Sort by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_n_items = [pred.iid for pred in predictions[:n]]
        
        return self.books_df[self.books_df['ISBN'].isin(top_n_items)]
    
    def get_hybrid_recommendations(self, user_id, n=10):
        """Get hybrid recommendations combining collaborative and content-based approaches."""
        # Get collaborative filtering recommendations
        cf_recs = self.get_collaborative_recommendations(user_id, n=n*2)  # Get more than needed to have room for blending
        
        # Get user's highly rated books
        user_ratings = self.ratings_df[self.ratings_df['User-ID'] == user_id]
        if len(user_ratings) == 0:
            print(f"No ratings found for user {user_id}. Using collaborative filtering only.")
            return cf_recs.head(n)  # No ratings, return just CF recommendations
            
        # Get the user's highest rated book
        try:
            # Check if we have any content-based indices at all
            if not hasattr(self, 'content_book_indices') or len(self.content_book_indices) == 0:
                print("No content book indices available. Using collaborative filtering only.")
                return cf_recs.head(n)
                
            # Try to find a book that the user has rated that is also in our content model
            content_book_found = False
            book_index = None
            
            # Check all user-rated books, starting from highest rated
            for _, row in user_ratings.sort_values('Book-Rating', ascending=False).iterrows():
                isbn = row['ISBN']
                # Try to find the book in our books_df
                matching_books = self.books_df[self.books_df['ISBN'] == isbn]
                if len(matching_books) > 0:
                    book_idx = matching_books.index[0]
                    # Check if this book is in our content model
                    if book_idx in self.content_book_indices:
                        book_index = book_idx
                        content_book_found = True
                        break
            
            # If no user-rated book is in the content model, try finding a book by the same author
            if not content_book_found:
                # Get authors of books the user has rated
                user_rated_isbns = user_ratings['ISBN'].tolist()
                user_book_authors = set()
                for isbn in user_rated_isbns:
                    author_books = self.books_df[self.books_df['ISBN'] == isbn]
                    if len(author_books) > 0:
                        user_book_authors.add(author_books['Book-Author'].values[0])
                
                # Try to find a book by the same author that is in the content model
                for author in user_book_authors:
                    author_books = self.books_df[
                        (self.books_df['Book-Author'] == author) & 
                        (self.books_df.index.isin(self.content_book_indices.keys()))
                    ]
                    if len(author_books) > 0:
                        book_index = author_books.index[0]
                        content_book_found = True
                        print(f"Found book by same author ({author}) in content model.")
                        break
            
            # If still nothing found, use a random book from the content model
            if not content_book_found:
                # Use a random book from the content model
                book_index = list(self.content_book_indices.keys())[0]
                book_info = self.books_df.loc[book_index]
                print(f"No books matching user preferences found in content model. Using a random book: {book_info['Book-Title']} by {book_info['Book-Author']}")
            
            # Get content-based recommendations
            cb_recs = self.get_content_based_recommendations(book_index, n=n*2)
            
            # Combine recommendations (weighted approach)
            cf_scores = {}
            for _, row in cf_recs.iterrows():
                cf_scores[row['ISBN']] = self.hybrid_weight
                
            cb_scores = {}
            for _, row in cb_recs.iterrows():
                cb_scores[row['ISBN']] = 1 - self.hybrid_weight
                
            # Combine scores
            hybrid_scores = defaultdict(float)
            for isbn, score in cf_scores.items():
                hybrid_scores[isbn] += score
                
            for isbn, score in cb_scores.items():
                hybrid_scores[isbn] += score
                
            # Sort by combined score
            sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            top_n_items = [item[0] for item in sorted_recommendations[:n]]
            
            return self.books_df[self.books_df['ISBN'].isin(top_n_items)]
        
        except (IndexError, ValueError, KeyError) as e:
            print(f"Error getting hybrid recommendations: {e}. Falling back to collaborative filtering.")
            return cf_recs.head(n)
    
    def optimize_hybrid_weight(self, user_sample=100, k=10):
        """Find optimal weight for hybrid recommendations."""
        # Check if content model is available
        if not hasattr(self, 'content_book_indices') or len(self.content_book_indices) == 0:
            print("Warning: No content-based model indices available. Optimization will use collaborative filtering only.")
            print("Setting optimal hybrid weight to 1.0 (collaborative filtering only).")
            self.hybrid_weight = 1.0
            return 1.0, pd.DataFrame([{'weight': 1.0, 'precision': 0, 'recall': 0, 'f1': 0}])
            
        # Sample users who have at least 3 ratings
        user_ratings = self.ratings_df['User-ID'].value_counts()
        eligible_users = user_ratings[user_ratings >= 3].index.tolist()
        
        if len(eligible_users) > user_sample:
            np.random.seed(42)
            sampled_users = np.random.choice(eligible_users, user_sample, replace=False)
        else:
            sampled_users = eligible_users
        
        weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        results = []
        
        # Track hybrid recommendation success rate
        hybrid_success_count = 0
        total_attempts = 0
        
        for weight in weights:
            self.hybrid_weight = weight
            precision_sum = 0
            recall_sum = 0
            f1_sum = 0
            count = 0
            
            for user_id in sampled_users:
                total_attempts += 1
                
                # Split user's ratings into train/test
                user_ratings = self.ratings_df[self.ratings_df['User-ID'] == user_id]
                train_ratings, test_ratings = train_test_split(user_ratings, test_size=0.3, random_state=42)
                
                # Get highly rated items from test set (ground truth)
                relevant_items = set(test_ratings[test_ratings['Book-Rating'] >= 8]['ISBN'])
                
                if len(relevant_items) == 0:
                    continue
                
                # Temporarily update ratings df to contain only training data
                original_ratings = self.ratings_df.copy()
                self.ratings_df = pd.concat([self.ratings_df[self.ratings_df['User-ID'] != user_id], train_ratings])
                
                # Retrain CF model on this modified data
                reader = Reader(rating_scale=(1, 10))
                data = Dataset.load_from_df(self.ratings_df[['User-ID', 'ISBN', 'Book-Rating']], reader)
                trainset = data.build_full_trainset()
                self.cf_algo.fit(trainset)
                
                # Get hybrid recommendations (suppress detailed error messages)
                try:
                    # Capture and suppress standard output temporarily to avoid printing errors for each user
                    import sys
                    from io import StringIO
                    
                    # Save the original stdout
                    original_stdout = sys.stdout
                    
                    # Redirect stdout to capture output
                    sys.stdout = StringIO()
                    
                    # Try to get hybrid recommendations
                    hybrid_recs = self.get_hybrid_recommendations(user_id, n=k)
                    hybrid_success = True
                    hybrid_success_count += 1
                    
                    # Restore stdout
                    sys.stdout = original_stdout
                except Exception as e:
                    # Restore stdout if an exception occurs
                    sys.stdout = original_stdout
                    print(f"Error during optimization for user {user_id}: {str(e)}")
                    hybrid_success = False
                
                recommended_items = set(hybrid_recs['ISBN'])
                
                # Calculate metrics
                n_rel_and_rec = len(relevant_items.intersection(recommended_items))
                precision = n_rel_and_rec / k if k > 0 else 0
                recall = n_rel_and_rec / len(relevant_items) if len(relevant_items) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                count += 1
                
                # Restore original ratings
                self.ratings_df = original_ratings
            
            # Calculate average metrics
            avg_precision = precision_sum / count if count > 0 else 0
            avg_recall = recall_sum / count if count > 0 else 0
            avg_f1 = f1_sum / count if count > 0 else 0
            
            results.append({
                'weight': weight,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            })
            
            print(f"Weight {weight}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")
        
        # Find optimal weight based on F1 score
        results_df = pd.DataFrame(results)
        optimal_weight = results_df.loc[results_df['f1'].idxmax()]['weight']
        self.hybrid_weight = optimal_weight
        
        # Report hybrid success rate
        success_rate = (hybrid_success_count / total_attempts) * 100 if total_attempts > 0 else 0
        print(f"\nHybrid recommendation success rate: {success_rate:.2f}% ({hybrid_success_count}/{total_attempts})")
        
        print(f"\nOptimal hybrid weight: {optimal_weight}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['weight'], results_df['precision'], 'o-', label='Precision@K')
        plt.plot(results_df['weight'], results_df['recall'], 's-', label='Recall@K')
        plt.plot(results_df['weight'], results_df['f1'], '^-', label='F1@K')
        plt.xlabel('CF Weight (1-weight for CB)')
        plt.ylabel('Score')
        plt.title(f'Hybrid Weight Optimization (k={k})')
        plt.legend()
        plt.grid(True)
        plt.savefig('weight_optimization.png')
        plt.close()
        
        return optimal_weight, results_df

# Example usage
if __name__ == "__main__":
    # Paths to dataset files
    books_path = 'Recommender/Books.csv'
    ratings_path = 'Recommender/Ratings.csv'
    users_path = 'Recommender/Users.csv'
    
    # Create and train the recommender
    recommender = BookRecommender(books_path, ratings_path, users_path)
    recommender.load_data()
    
    # Train models
    recommender.train_content_based_model()
    recommender.train_collaborative_filtering(algorithm='svd')
    
    # Evaluate models
    print("\nEvaluating collaborative filtering model:")
    recommender.evaluate_collaborative_filtering()
    
    print("\nEvaluating metrics at different k values:")
    recommender.evaluate_metrics_at_k(k_values=[5, 10, 15, 20])
    
    print("\nOptimizing hybrid weight:")
    recommender.optimize_hybrid_weight(user_sample=50, k=10)
    
    # Get recommendations for a sample user
    sample_user_id = recommender.ratings_df['User-ID'].value_counts().index[0]
    print(f"\nHybrid recommendations for user {sample_user_id}:")
    hybrid_recs = recommender.get_hybrid_recommendations(sample_user_id, n=10)
    print(hybrid_recs[['ISBN', 'Book-Title', 'Book-Author']]) 