from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import traceback
from hybrid_recommender import HybridBookRecommender

app = Flask(__name__)

# Initialize the recommender system
MODEL_PATH = 'Recommender/book_recommender_model.pkl'
recommender = HybridBookRecommender()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/search')
def search():
    """Search for books by title"""
    try:
        query = request.args.get('query', '').lower()
        if not query:
            return jsonify([])
            
        # Get matching book titles
        matching_titles = [
            title for title in recommender.merged_df['Book-Title'].unique()
            if query in title.lower()
        ]
        matching_titles.sort()
        
        # Limit results to top 10
        return jsonify(matching_titles[:10])
    except Exception as e:
        app.logger.error(f"Error in search: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/recommend')
def recommend():
    """Get recommendations for a book"""
    try:
        book_title = request.args.get('book', '')
        n = int(request.args.get('n', 10))
        
        if not book_title:
            return jsonify({'error': 'No book title provided'})
        
        # Get recommendations
        recommended_books = recommender.get_hybrid_recommendations(book_title, n=n)
        
        # Get book details
        recommendations_with_details = []
        
        # Add the input book at the beginning
        if book_title in recommender.merged_df['Book-Title'].values:
            input_book_data = recommender.merged_df[recommender.merged_df['Book-Title'] == book_title].iloc[0]
            recommendations_with_details.append({
                'title': input_book_data['Book-Title'],
                'author': input_book_data['Book-Author'],
                'year': str(input_book_data['Year-Of-Publication']),
                'isbn': input_book_data['ISBN'],
                'publisher': input_book_data['Publisher'],
                'cover': input_book_data.get('Image-URL-L', ''),
                'is_input': True
            })
        
        # Add recommended books
        for title in recommended_books:
            if title == book_title:
                continue  # Skip if it's the input book
                
            book_data = recommender.merged_df[recommender.merged_df['Book-Title'] == title].iloc[0]
            recommendations_with_details.append({
                'title': book_data['Book-Title'],
                'author': book_data['Book-Author'],
                'year': str(book_data['Year-Of-Publication']),
                'isbn': book_data['ISBN'],
                'publisher': book_data['Publisher'],
                'cover': book_data.get('Image-URL-L', ''),
                'is_input': False
            })
            
        return jsonify({
            'input_book': book_title,
            'recommendations': recommendations_with_details
        })
    except Exception as e:
        app.logger.error(f"Error in recommend: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})

@app.route('/popular')
def popular():
    """Get popular books"""
    try:
        n = int(request.args.get('n', 10))
        
        # Get popular books
        popular_books = recommender.get_popular_recommendations(n=n).index.tolist()
        
        # Get book details
        popular_with_details = []
        for title in popular_books:
            book_data = recommender.merged_df[recommender.merged_df['Book-Title'] == title].iloc[0]
            popular_with_details.append({
                'title': book_data['Book-Title'],
                'author': book_data['Book-Author'],
                'year': str(book_data['Year-Of-Publication']),
                'isbn': book_data['ISBN'],
                'publisher': book_data['Publisher'],
                'cover': book_data.get('Image-URL-L', '')
            })
            
        return jsonify(popular_with_details)
    except Exception as e:
        app.logger.error(f"Error in popular: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/compare')
def compare():
    """Compare different recommendation approaches"""
    try:
        book_title = request.args.get('book', '')
        n = int(request.args.get('n', 5))
        
        if not book_title:
            return jsonify({'error': 'No book title provided'})
        
        # Get recommendations from each approach
        content_recs = recommender.get_content_recommendations(book_title, n=n)
        collab_recs = recommender.get_collaborative_recommendations(book_title, n=n)
        hybrid_recs = recommender.get_hybrid_recommendations(book_title, n=n)
        
        # Convert to list format
        content_books = content_recs.index.tolist() if not content_recs.empty else []
        collab_books = collab_recs.index.tolist() if not collab_recs.empty else []
        
        return jsonify({
            'input_book': book_title,
            'content_based': content_books,
            'collaborative': collab_books,
            'hybrid': hybrid_recs
        })
    except Exception as e:
        app.logger.error(f"Error in compare: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    try:
        print("Starting Book Recommendation System...")
        
        # Load or build the model
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}")
            recommender.load_model(MODEL_PATH)
        else:
            print("Building recommendation model...")
            # Load data
            recommender.load_data(
                books_path='Recommender/Books.csv',
                ratings_path='Recommender/Ratings.csv',
                users_path='Recommender/Users.csv'
            )
            
            # Preprocess data and build models
            print("Preprocessing data...")
            recommender.preprocess_data(min_book_ratings=20, min_user_ratings=5)
            
            print("Building content-based model...")
            recommender.build_content_based_model()
            
            print("Building collaborative filtering model...")
            recommender.build_collaborative_model(n_components=100)
            
            # Save model
            print("Saving model...")
            recommender.save_model(MODEL_PATH)
        
        # Create templates directory if it doesn't exist
        os.makedirs('Recommender/templates', exist_ok=True)
        
        print("Starting web server...")
        # Run the app
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        traceback.print_exc() 