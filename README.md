# Hybrid Book Recommendation System

A comprehensive book recommendation system that combines multiple recommendation techniques to deliver high-quality book suggestions.

## Features

- **Hybrid Recommendation Engine** - Combines content-based filtering, collaborative filtering, and popularity-based methods
- **Web Interface** - User-friendly interface to search and explore book recommendations
- **Method Comparison** - View and compare recommendations from different approaches
- **Popular Books** - Discover trending books based on user ratings

## Recommendation Approaches

This system implements three primary recommendation approaches:

1. **Content-Based Filtering**: Recommends books similar to those a user has liked before, based on book attributes (author, publisher, year)

2. **Collaborative Filtering**: Recommends books based on user behavior and preferences
   - Uses matrix factorization (SVD) to identify latent factors in the user-book interaction data
   - Calculates item-item similarity to find books frequently rated similarly

3. **Popularity-Based**: Recommends popular books that have high ratings and many reviews

4. **Hybrid Approach**: Combines the strengths of all three methods with configurable weights

## Dataset

The system uses the Book-Crossing dataset which includes:
- 278,858 users
- 271,379 books
- 1,149,780 ratings

## Installation

### Prerequisites
- Python 3.6+
- pandas
- numpy
- scikit-learn
- Flask
- pickle

### Setup

1. Clone the repository
```
git clone https://github.com/yourusername/book-recommender.git
cd book-recommender
```

2. Install required packages
```
pip install -r requirements.txt
```

3. Place the dataset files in the `Recommender` directory:
   - Books.csv
   - Ratings.csv
   - Users.csv

## Usage

### Running the Web Application

1. Start the Flask application:
```
python Recommender/recommender_app.py
```

2. Open your browser and go to `http://localhost:5000`

### Using the Recommendation API

The system provides several API endpoints:

- `/search?query=<book_title>` - Search for books by title
- `/recommend?book=<book_title>&n=<num_recommendations>` - Get hybrid recommendations
- `/popular?n=<num_books>` - Get popular books
- `/compare?book=<book_title>&n=<num_recommendations>` - Compare different recommendation approaches

### Running the Standalone Recommender

You can also use the recommender directly in Python:

```python
from Recommender.hybrid_recommender import HybridBookRecommender

# Initialize the recommender
recommender = HybridBookRecommender(
    content_weight=0.3,
    collab_weight=0.5,
    popular_weight=0.2
)

# Load and preprocess data
recommender.load_data(
    books_path='Recommender/Books.csv',
    ratings_path='Recommender/Ratings.csv',
    users_path='Recommender/Users.csv'
)
recommender.preprocess_data()

# Build models
recommender.build_content_based_model()
recommender.build_collaborative_model()

# Get recommendations
recommendations = recommender.get_hybrid_recommendations("The Da Vinci Code", n=10)
print(recommendations)
```

## Customization

You can adjust the hybrid model weights to emphasize different recommendation approaches:

```python
recommender = HybridBookRecommender(
    content_weight=0.5,  # Increase content-based influence
    collab_weight=0.3,   # Decrease collaborative filtering influence
    popular_weight=0.2   # Keep popularity influence the same
)
```

## Model Persistence

The system automatically saves the trained model to avoid retraining:

```python
# Save model
recommender.save_model('book_recommender_model.pkl')

# Load model
recommender.load_model('book_recommender_model.pkl')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 