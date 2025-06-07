#!/usr/bin/env python3
"""
Quick Demo of Hybrid Recommender System
This script provides a quick test of the system with existing data
"""

import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def quick_demo():
    """Quick demonstration of the hybrid recommender system"""
    
    print("="*60)
    print("HYBRID RECOMMENDER SYSTEM - QUICK DEMO")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if we can import our modules
    try:
        from hybrid_recommender_system import HybridRecommenderSystem
        from evaluation_framework import RecommenderEvaluator
        print("✓ Successfully imported hybrid recommender modules")
    except ImportError as e:
        print(f"✗ Error importing modules: {e}")
        print("Please ensure all files are in the same directory")
        return
    
    # Check for data files
    data_files = {
        'books': 'data/Books.csv',
        'ratings_1': 'Recommender/Ratings.csv',
        'ratings_2': 'data/Ratings.csv',  # Alternative location
        'users_1': 'Recommender/Users.csv',
        'users_2': 'data/Users.csv'  # Alternative location
    }
    
    print("Checking for data files...")
    available_files = {}
    for name, path in data_files.items():
        if os.path.exists(path):
            print(f"✓ Found: {path}")
            available_files[name] = path
        else:
            print(f"✗ Not found: {path}")
    
    # Determine which files to use
    books_path = available_files.get('books')
    ratings_path = available_files.get('ratings_1') or available_files.get('ratings_2')
    users_path = available_files.get('users_1') or available_files.get('users_2')
    
    if not books_path or not ratings_path:
        print("\n✗ Missing required data files (Books.csv and Ratings.csv)")
        print("Please ensure you have:")
        print("  - data/Books.csv")
        print("  - Recommender/Ratings.csv (or data/Ratings.csv)")
        return
    
    print(f"\nUsing data files:")
    print(f"  Books: {books_path}")
    print(f"  Ratings: {ratings_path}")
    if users_path:
        print(f"  Users: {users_path}")
    print()
    
    # Initialize recommender system
    print("Initializing hybrid recommender system...")
    try:
        recommender = HybridRecommenderSystem(random_state=42)
        print("✓ Recommender system initialized")
    except Exception as e:
        print(f"✗ Error initializing recommender: {e}")
        return
    
    # Load data
    print("Loading data...")
    try:
        recommender.load_data(
            books_path=books_path,
            ratings_path=ratings_path,
            users_path=users_path
        )
        print("✓ Data loaded successfully")
        print(f"  Books: {len(recommender.books_df):,}")
        print(f"  Ratings: {len(recommender.ratings_df):,}")
        print(f"  Users: {recommender.ratings_df['User-ID'].nunique():,}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Build a simple model for demonstration
    print("\nBuilding models (simplified for demo)...")
    try:
        # Build content-based model
        print("  Building content-based model...")
        recommender.build_content_based_models()
        print("  ✓ Content-based models built")
        
        # Build one collaborative model
        print("  Building collaborative filtering model...")
        recommender.build_collaborative_models()
        print("  ✓ Collaborative filtering models built")
        
        # Build popularity model
        print("  Building popularity model...")
        recommender.build_popularity_model()
        print("  ✓ Popularity model built")
        
    except Exception as e:
        print(f"✗ Error building models: {e}")
        return
    
    # Demonstrate recommendations
    print("\nDemonstrating recommendations...")
    
    # Get sample data
    sample_users = recommender.ratings_df['User-ID'].unique()[:3]
    sample_books = recommender.books_df['ISBN'].iloc[:3].tolist()
    
    if len(sample_users) > 0 and len(sample_books) > 0:
        sample_user = sample_users[0]
        sample_isbn = sample_books[0]
        sample_book = recommender.books_df[recommender.books_df['ISBN'] == sample_isbn].iloc[0]
        
        print(f"\nSample User: {sample_user}")
        print(f"Sample Book: '{sample_book['Book-Title']}' by {sample_book['Book-Author']}")
        print()
        
        # Content-based recommendations
        try:
            print("Content-based recommendations:")
            content_recs = recommender.get_content_recommendations(sample_isbn, 'tfidf', 3)
            for i, rec in enumerate(content_recs, 1):
                print(f"  {i}. {rec['title']} by {rec['author']}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # Collaborative recommendations
        try:
            print("\nCollaborative filtering recommendations:")
            collab_recs = recommender.get_collaborative_recommendations(sample_user, 'svd', 3)
            for i, rec in enumerate(collab_recs, 1):
                print(f"  {i}. {rec['title']} by {rec['author']}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # Popularity recommendations
        try:
            print("\nPopularity-based recommendations:")
            pop_recs = recommender.get_popularity_recommendations(3)
            for i, rec in enumerate(pop_recs, 1):
                print(f"  {i}. {rec['title']} by {rec['author']}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # Hybrid recommendations
        try:
            print("\nHybrid recommendations:")
            hybrid_recs = recommender.get_hybrid_recommendations(
                user_id=sample_user, 
                isbn=sample_isbn, 
                n=3, 
                strategy='weighted'
            )
            for i, rec in enumerate(hybrid_recs, 1):
                print(f"  {i}. {rec['title']} by {rec['author']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Quick evaluation (limited)
    print("\nRunning quick evaluation...")
    try:
        evaluator = RecommenderEvaluator(recommender)
        
        # Prepare small evaluation dataset
        evaluator.prepare_evaluation_data(test_size=0.2, min_interactions=5)
        
        # Run accuracy evaluation only (faster)
        accuracy_results = evaluator.evaluate_accuracy_metrics(['svd'])
        
        print("✓ Quick evaluation completed")
        print("\nAccuracy Results:")
        for model, metrics in accuracy_results.items():
            print(f"  {model.upper()}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
    
    print("\n" + "="*60)
    print("QUICK DEMO COMPLETED!")
    print("="*60)
    print("\nTo run the full system with comprehensive evaluation:")
    print("  python main_recommender_pipeline.py")
    print("\nTo explore the system interactively:")
    print("  python -i quick_demo.py")
    print("  >>> # Now you can interact with the 'recommender' object")

if __name__ == "__main__":
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc() 