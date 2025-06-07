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

# Book Data Processing and Database Import

This project exports preprocessed book data from the hybrid recommender system, enriches it with Google Books API data, and imports it into a MySQL database.

## Requirements

- Python 3.6+
- MySQL server
- Required Python packages: `pandas`, `sqlalchemy`, `pymysql`, `requests`

## Setup

1. Install required packages:
```
pip install pandas sqlalchemy pymysql requests
```

2. Configure your MySQL connection in `init_database.py` and `import_books_into_database.py`:
```python
engine = create_engine('mysql+pymysql://root:12345678@localhost:3306/myLib')
```

3. Add your Google Books API key in `import_books_into_database.py`:
```python
GOOGLE_API_KEY = "YOUR_API_KEY"  # Replace with your actual Google API key
```

## Usage

### Option 1: Run the complete process

```
python process_and_import_books.py
```

This will:
1. Initialize the database and create the books table if needed
2. Export preprocessed books from the recommender system to CSV file
3. Import the books into the database with additional data from Google Books API

### Option 2: Run individual steps

Initialize the database:
```
python init_database.py
```

Export preprocessed books to CSV:
```
python export_preprocessed_books.py
```

Import books from CSV to database:
```
python import_books_into_database.py
```

## Files

- `init_database.py`: Creates the myLib database and books table if they don't exist
- `export_preprocessed_books.py`: Exports preprocessed books from the recommender system to CSV
- `import_books_into_database.py`: Imports books from CSV to database with Google Books API data
- `process_and_import_books.py`: Main script that runs all steps in sequence
- `preprocessing_books.csv`: CSV file containing exported book data

## Database Schema

The books table has the following structure:

```sql
CREATE TABLE books (
  book_id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  author VARCHAR(255),
  publisher VARCHAR(255),
  publication_year INT,
  isbn VARCHAR(255) NOT NULL UNIQUE,
  genre VARCHAR(255),
  descriptions TEXT,
  cover_image_url VARCHAR(255),
  quantity INT NOT NULL,
  available_quantity INT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  preview_link VARCHAR(255),
  updated_at DATETIME(6)
);
```

## Notes

- The script generates random values for `quantity` (1-10) and `available_quantity` (0-quantity)
- The Google Books API has rate limits. The script includes delays to avoid hitting these limits.
- For books not found in the Google Books API, only basic information will be imported. 