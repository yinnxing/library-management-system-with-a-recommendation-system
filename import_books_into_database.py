import pandas as pd
import requests
import time
import random
from sqlalchemy import create_engine, text
from datetime import datetime
import re

# Google Books API key - you need to get your own API key from Google Cloud Console
# Visit https://console.cloud.google.com/, create a project, enable the Books API,
# and generate an API key under "APIs & Services" > "Credentials"
GOOGLE_API_KEY = "AIzaSyAi2Qxrpo88GPIdQt22XQBb-fXtLI9yfDo"  # Replace with your actual Google API key

def clean_text(text):
    """Clean text by removing HTML tags and extra whitespace"""
    if not text:
        return None
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def truncate_text(text, max_length=65000):
    """Truncate text to maximum length to avoid database errors"""
    if not text:
        return None
    if len(text) > max_length:
        print(f"Warning: Truncating text from {len(text)} to {max_length} characters")
        return text[:max_length - 3] + "..."
    return text

def fetch_book_details(isbn, title, author):
    """Fetch book details from Google Books API"""
    # Construct the query - try ISBN first, then title and author
    query = f"isbn:{isbn}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&key={GOOGLE_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # If no results with ISBN, try with title and author
        if 'items' not in data or len(data['items']) == 0:
            query = f"intitle:{title}"
            if author:
                query += f"+inauthor:{author}"
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}&key={GOOGLE_API_KEY}"
            response = requests.get(url)
            data = response.json()
        
        # If still no results, return None
        if 'items' not in data or len(data['items']) == 0:
            return None
        
        # Get the first item
        book_info = data['items'][0]['volumeInfo']
        
        # Extract relevant information
        details = {
            'genre': truncate_text(', '.join(book_info.get('categories', [])) if 'categories' in book_info else None, 255),
            'descriptions': truncate_text(clean_text(book_info.get('description'))),
            'cover_image_url': truncate_text(book_info.get('imageLinks', {}).get('thumbnail'), 255),
            'preview_link': truncate_text(book_info.get('previewLink'), 255)
        }
        
        return details
    
    except Exception as e:
        print(f"Error fetching details for ISBN {isbn}: {str(e)}")
        return None

def import_books_to_database():
    """Import books from CSV to MySQL database with Google Books API data"""
    # Load the preprocessed books
    try:
        books_df = pd.read_csv('preprocessing_books.csv')
        print(f"Loaded {len(books_df)} books from CSV")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return
    
    # Connect to the database
    try:
        engine = create_engine('mysql+pymysql://root:12345678@localhost:3306/myLib')
        connection = engine.connect()
        print("Connected to database")
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return
    
    # Check if books table exists
    try:
        connection.execute(text("SELECT 1 FROM books LIMIT 1"))
        print("Books table exists")
    except:
        print("Books table doesn't exist or is not accessible")
        return
    
    # Process each book
    books_added = 0
    books_skipped = 0
    
    for index, book in books_df.iterrows():
        # Check if book already exists in database
        try:
            result = connection.execute(
                text("SELECT book_id FROM books WHERE isbn = :isbn"),
                {"isbn": book['isbn']}
            ).fetchone()
            
            if result:
                books_skipped += 1
                if index % 50 == 0:
                    print(f"Skipped existing book: {book['title']} (ISBN: {book['isbn']})")
                continue
        except Exception as e:
            print(f"Error checking for existing book {book['isbn']}: {str(e)}")
            continue
        
        # Fetch additional details from Google Books API
        print(f"Fetching details for: {book['title']} (ISBN: {book['isbn']})")
        api_details = fetch_book_details(book['isbn'], book['title'], book['author'])
        
        # Generate random quantities
        quantity = random.randint(1, 10)
        available_quantity = random.randint(0, quantity)
        
        # Prepare book data for insertion with default values for all fields
        book_data = {
            "title": truncate_text(book['title'], 255),
            "author": truncate_text(book['author'] if not pd.isna(book['author']) else "Unknown Author", 255),
            "publisher": truncate_text(book['publisher'] if not pd.isna(book['publisher']) else "Unknown Publisher", 255),
            "publication_year": int(book['publication_year']) if not pd.isna(book['publication_year']) else None,
            "isbn": book['isbn'],
            "genre": None,
            "descriptions": None,
            "cover_image_url": None,
            "preview_link": None,
            "quantity": quantity,
            "available_quantity": available_quantity,
            "updated_at": datetime.now()
        }
        
        # Add API details if available
        if api_details:
            # Only update fields that are not None in api_details
            for key, value in api_details.items():
                if value is not None:
                    book_data[key] = value
        
        # Insert into database
        try:
            query = text("""
                INSERT INTO books (
                    title, author, publisher, publication_year, isbn, 
                    genre, descriptions, cover_image_url, quantity, 
                    available_quantity, updated_at, preview_link
                ) VALUES (
                    :title, :author, :publisher, :publication_year, :isbn,
                    :genre, :descriptions, :cover_image_url, :quantity,
                    :available_quantity, :updated_at, :preview_link
                )
            """)
            
            connection.execute(query, book_data)
            books_added += 1
            
            if index % 10 == 0:
                print(f"Added book: {book['title']} (ISBN: {book['isbn']})")
            
            # Sleep to avoid hitting API rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error inserting book {book['isbn']}: {str(e)}")
            # Print the actual data that caused the error for debugging
            print(f"Book data: {book_data}")
            
            # Try again without the descriptions field if it's a data too long error
            if "Data too long for column" in str(e):
                try:
                    print("Retrying without the problematic field...")
                    
                    # Identify which field is causing the issue
                    error_field = None
                    if "descriptions" in str(e):
                        error_field = "descriptions"
                    elif "genre" in str(e):
                        error_field = "genre"
                    elif "cover_image_url" in str(e):
                        error_field = "cover_image_url"
                    elif "preview_link" in str(e):
                        error_field = "preview_link"
                    
                    if error_field:
                        print(f"Removing {error_field} field and retrying...")
                        book_data[error_field] = None
                    
                    connection.execute(query, book_data)
                    books_added += 1
                    print(f"Successfully added book without the problematic field: {book['title']} (ISBN: {book['isbn']})")
                except Exception as retry_error:
                    print(f"Retry failed: {str(retry_error)}")
    
    # Commit changes and close connection
    connection.commit()
    connection.close()
    
    print(f"\nImport complete: {books_added} books added, {books_skipped} books skipped")

if __name__ == "__main__":
    import_books_to_database() 