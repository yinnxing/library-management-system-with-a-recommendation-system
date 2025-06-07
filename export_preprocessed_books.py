import pandas as pd
import pickle
import os
import numpy as np
from Recommender.hybrid_recommender import HybridBookRecommender

def export_preprocessed_books():
    # Check if model file exists, otherwise create a new recommender and preprocess data
    model_path = 'Recommender/book_recommender_model.pkl'
    
    if os.path.exists(model_path):
        # Load existing model
        print("Loading existing recommender model...")
        recommender = HybridBookRecommender()
        recommender.load_model(model_path)
    else:
        # Create new recommender and process data
        print("Creating new recommender and preprocessing data...")
        recommender = HybridBookRecommender()
        
        # Load data
        recommender.load_data(
            books_path='Recommender/Books.csv',
            ratings_path='Recommender/Ratings.csv',
            users_path='Recommender/Users.csv'
        )
        
        # Preprocess data
        recommender.preprocess_data(
            min_book_ratings=50,
            min_user_ratings=3,
            verbose=True
        )
        
        # Save the model for future use
        recommender.save_model(model_path)
    
    # Extract unique books from the merged dataframe
    if recommender.merged_df is not None:
        books_df = recommender.merged_df[['ISBN', 'Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']].drop_duplicates('ISBN')
        
        # Clean data
        books_df = books_df.fillna({
            'Book-Title': 'Unknown Title',
            'Book-Author': 'Unknown Author',
            'Publisher': 'Unknown Publisher'
        })
        
        # Handle publication year - ensure it's numeric and within reasonable range
        books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
        # Replace invalid years (too old or future) with NaN
        books_df.loc[books_df['Year-Of-Publication'] < 1800, 'Year-Of-Publication'] = np.nan
        books_df.loc[books_df['Year-Of-Publication'] > 2023, 'Year-Of-Publication'] = np.nan
        
        # Ensure ISBN is a string and not NaN
        books_df['ISBN'] = books_df['ISBN'].astype(str)
        books_df = books_df[books_df['ISBN'] != 'nan']  # Remove rows with NaN ISBNs
        
        # Rename columns to match database schema
        books_df = books_df.rename(columns={
            'ISBN': 'isbn',
            'Book-Title': 'title',
            'Book-Author': 'author',
            'Publisher': 'publisher',
            'Year-Of-Publication': 'publication_year'
        })
        
        # Export to CSV
        output_path = 'preprocessing_books_temp.csv'
        books_df.to_csv(output_path, index=False)
        print(f"Exported {len(books_df)} unique books to {output_path}")
        return output_path
    else:
        print("No preprocessed data available")
        return None

if __name__ == "__main__":
    export_preprocessed_books() 