
# Revised script: import_books_to_database with proper transactions and detailed debugging
import pandas as pd
import requests
import time
import random
import logging
import re
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Google Books API key
GOOGLE_API_KEY = "YOUR_API_KEY"  # Replace with your actual Google API key


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
        logger.warning(f"Truncating text from {len(text)} to {max_length} chars")
        return text[:max_length - 3] + "..."
    return text


def fetch_book_details(isbn, title, author):
    """Fetch detailed book info from Google Books API"""
    query = f"isbn:{isbn}"
    base_url = "https://www.googleapis.com/books/v1/volumes"

    def request_api(q):
        url = f"{base_url}?q={q}&key={GOOGLE_API_KEY}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    try:
        data = request_api(query)
        if not data.get('items'):
            # fallback to title/author search
            q2 = f"intitle:{title}" + (f"+inauthor:{author}" if author else "")
            data = request_api(q2)
            if not data.get('items'):
                return None

        info = data['items'][0]['volumeInfo']
        # Choose highest resolution image
        img_links = info.get('imageLinks', {})
        cover_url = img_links.get('extraLarge') or img_links.get('large') or img_links.get('thumbnail')

        details = {
            'genre': truncate_text(', '.join(info.get('categories', [])), 255) if info.get('categories') else None,
            'descriptions': truncate_text(clean_text(info.get('description'))),
            'cover_image_url': truncate_text(cover_url, 255),
            'preview_link': truncate_text(info.get('previewLink'), 255)
        }
        return details

    except Exception as ex:
        logger.error(f"Error fetching details for {isbn} - {title}: {ex}")
        return None


def import_books_to_database(csv_path, db_url):
    """Import books from CSV to MySQL database with Google Books data"""
    # Load data
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        logger.error(f"Cannot read CSV: {e}")
        return

    # Create engine
    engine = create_engine(db_url)

    # We'll use a single transaction block for all inserts
    with engine.begin() as conn:
        for idx, row in df.iterrows():
            isbn = row['isbn']
            # Skip existing
            exists = conn.execute(
                text("SELECT 1 FROM books WHERE isbn=:isbn LIMIT 1"), {'isbn': isbn}
            ).fetchone()
            if exists:
                logger.debug(f"Skipping existing ISBN {isbn}")
                continue

            # Fetch API details
            logger.info(f"Fetching for ISBN {isbn}")
            details = fetch_book_details(isbn, row['title'], row.get('author'))

            # Prepare data
            qty = random.randint(1, 10)
            avail = random.randint(0, qty)
            data = {
                'title': truncate_text(row['title'], 255),
                'author': truncate_text(row['author'] if pd.notna(row.get('author')) else 'Unknown Author', 255),
                'publisher': truncate_text(row['publisher'] if pd.notna(row.get('publisher')) else 'Unknown Publisher', 255),
                'publication_year': int(row['publication_year']) if pd.notna(row.get('publication_year')) else None,
                'isbn': isbn,
                'genre': None,
                'descriptions': None,
                'cover_image_url': None,
                'preview_link': None,
                'quantity': qty,
                'available_quantity': avail,
                'updated_at': datetime.now()
            }
            if details:
                data.update({k:v for k,v in details.items() if v is not None})

            insert_sql = text(
                """
                INSERT INTO books
                (title, author, publisher, publication_year, isbn,
                 genre, descriptions, cover_image_url, quantity,
                 available_quantity, updated_at, preview_link)
                VALUES
                (:title, :author, :publisher, :publication_year, :isbn,
                 :genre, :descriptions, :cover_image_url, :quantity,
                 :available_quantity, :updated_at, :preview_link)
                """
            )
            try:
                conn.execute(insert_sql, data)
                logger.info(f"Inserted ISBN {isbn}")
                # avoid rate limit
                time.sleep(random.uniform(0.5, 1.5))
            except SQLAlchemyError as db_err:
                logger.error(f"DB error for ISBN {isbn}: {db_err}")
                # on data too long, retry without descriptions
                if 'Data too long' in str(db_err):
                    data['descriptions'] = None
                    try:
                        conn.execute(insert_sql, data)
                        logger.info(f"Inserted {isbn} after removing descriptions")
                    except Exception as retry_err:
                        logger.error(f"Retry failed for {isbn}: {retry_err}")
        logger.info("Import complete.")


if __name__ == '__main__':
    import_books_to_database('preprocessing_books.csv', 'mysql+pymysql://root:12345678@localhost:3306/myLib')
