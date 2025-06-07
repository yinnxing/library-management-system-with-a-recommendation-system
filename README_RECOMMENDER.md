# Hybrid Book Recommendation System

This project implements a hybrid book recommendation system that combines collaborative filtering and content-based approaches to provide personalized book recommendations.

## Features

- **Collaborative Filtering**: Uses matrix factorization (SVD) or k-nearest neighbors (KNN) to analyze user-item interactions
- **Content-Based Filtering**: Recommends books similar to ones a user has liked based on title, author, and publisher
- **Hybrid Approach**: Combines both methods for more robust recommendations
- **Evaluation Metrics**: Includes Precision@K, Recall@K, and F1@K for different K values
- **Optimization**: Automatically finds the optimal weight for hybrid recommendations
- **Memory Efficiency**: Handles large datasets by chunking and feature limiting
- **Interactive CLI**: Easy-to-use command-line interface for interacting with the system

## Dataset

The system uses the Book-Crossings dataset with three CSV files:
- `Books.csv`: Contains book metadata (ISBN, title, author, publication year, publisher, image URLs)
- `Ratings.csv`: Contains user ratings for books (User-ID, ISBN, Book-Rating)
- `Users.csv`: Contains user information (User-ID, Location, Age)

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To train the models and get recommendations for a specific user:

```bash
python run_recommender.py --train --recommend --user-id 12345
```

To get content-based recommendations for a specific book:

```bash
python run_recommender.py --train --recommend --book-isbn "0971880107"
```

### Advanced Usage

Evaluate the recommendation system with different K values:

```bash
python run_recommender.py --train --evaluate --k-values "5,10,15,20,25"
```

Optimize the hybrid recommendation weight:

```bash
python run_recommender.py --train --optimize
```

Control memory usage by limiting the number of books in the content model:

```bash
python run_recommender.py --train --recommend --max-books 2000
```

### Command-line Arguments

```
usage: run_recommender.py [-h] [--train] [--evaluate] [--optimize] [--recommend]
                         [--user-id USER_ID] [--book-isbn BOOK_ISBN] [--top-n TOP_N]
                         [--algorithm {svd,knn}] [--k-values K_VALUES] [--max-books MAX_BOOKS]

Book Recommendation System

optional arguments:
  -h, --help            show this help message and exit
  --train               Train the recommendation system
  --evaluate            Evaluate the recommendation system
  --optimize            Optimize hybrid weight
  --recommend           Get recommendations
  --user-id USER_ID     User ID to get recommendations for
  --book-isbn BOOK_ISBN Book ISBN to get content-based recommendations for
  --top-n TOP_N         Number of recommendations to return (default: 10)
  --algorithm {svd,knn} Collaborative filtering algorithm to use (default: svd)
  --k-values K_VALUES   Comma-separated list of k values for evaluation (default: 5,10,15,20)
  --max-books MAX_BOOKS Maximum number of books to include in content-based model (default: 5000)
```

## System Components

### 1. Data Loading and Preprocessing

- Loads and cleans data from CSV files
- Removes missing values and filters unpopular books (less than 10 reviews)
- Filters out inactive users (less than 3 reviews)
- Converts implicit feedback to explicit ratings

### 2. Content-Based Filtering

- Creates TF-IDF vectors from book metadata (title, author, publisher)
- Uses memory-efficient processing with chunking for large datasets
- Limits the number of books and features to prevent RAM overflow
- Computes cosine similarity matrix between books
- Recommends books similar to a reference book

### 3. Collaborative Filtering

- Uses SVD (Singular Value Decomposition) or KNN for matrix factorization
- Predicts user ratings for unrated books
- Recommends highest predicted rating books to users

### 4. Hybrid Recommendations

- Combines content-based and collaborative filtering recommendations
- Uses a weighted approach that can be optimized for best performance
- Balances between similar items and personalized preferences
- Gracefully handles cases where books aren't in the content model

### 5. Evaluation

- Splits data into training and testing sets
- Calculates Precision@K, Recall@K, and F1@K metrics
- Identifies optimal K values and hybrid weights

## Memory Management

The system implements several strategies to handle large datasets efficiently:

1. **Book Filtering**: Only popular books (10+ reviews) are used in recommendations
2. **User Filtering**: Only active users (3+ reviews) are considered
3. **Content Model Limiting**: Content-based model uses a configurable maximum number of books (default: 5000)
4. **Chunking**: Books are processed in smaller chunks to avoid memory overflow
5. **Feature Limiting**: TF-IDF vectorization limits features to the most relevant ones
6. **Data Type Optimization**: Uses float32 instead of float64 to reduce memory consumption

These optimizations allow the system to work with the full Book-Crossings dataset even on machines with limited RAM.

## Evaluation Results

The system evaluates recommendations using:

- **Precision@K**: Proportion of recommended items that are relevant
- **Recall@K**: Proportion of relevant items that are recommended
- **F1@K**: Harmonic mean of precision and recall

Evaluation results are displayed in the terminal and saved as plots in:
- `evaluation_metrics.png`: Metrics for different K values
- `weight_optimization.png`: Performance across different hybrid weights

## Examples

### Getting Recommendations for a User

```bash
python run_recommender.py --train --recommend --user-id 11676
```

This will:
1. Load and preprocess the dataset
2. Train both recommendation models
3. Display collaborative filtering recommendations
4. Display hybrid recommendations
5. Show the user's past ratings for context

### Finding Similar Books

```bash
python run_recommender.py --train --recommend --book-isbn "0971880107"
```

This will show books similar to the one with the specified ISBN based on content features.

## Technical Implementation

- The core recommendation engine is implemented in `book_recommender.py`
- The command-line interface is implemented in `run_recommender.py`
- The system uses scikit-surprise for collaborative filtering algorithms
- TF-IDF vectorization is used for content-based features
- The hybrid approach uses a weighted combination with optimizable weights

## Future Improvements

Potential enhancements for the system:
- Implement matrix factorization with deep learning models
- Add natural language processing for book descriptions
- Incorporate temporal dynamics to account for changing user preferences
- Create a web interface for easier interaction
- Add explainability features to help users understand recommendations 