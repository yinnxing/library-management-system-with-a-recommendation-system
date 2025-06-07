#!/usr/bin/env python3
"""
Initialize the myLib database and create the books table if it doesn't exist.
"""

from sqlalchemy import create_engine, text

def init_database():
    """Initialize the database and create tables if they don't exist"""
    # Database connection
    try:
        engine = create_engine('mysql+pymysql://root:12345678@localhost:3306/')
        connection = engine.connect()
        print("Connected to MySQL server")
        
        # Create database if it doesn't exist
        connection.execute(text("CREATE DATABASE IF NOT EXISTS myLib"))
        print("Ensured myLib database exists")
        
        # Switch to the myLib database
        connection.execute(text("USE myLib"))
        
        # Create books table if it doesn't exist
        create_books_table_query = text("""
        CREATE TABLE IF NOT EXISTS books (
          book_id INT AUTO_INCREMENT PRIMARY KEY,
          title VARCHAR(255) NOT NULL,
          author VARCHAR(255),
          publisher VARCHAR(255),
          publication_year INT,
          isbn VARCHAR(255) NOT NULL UNIQUE,
          genre VARCHAR(255),
          descriptions LONGTEXT,
          cover_image_url VARCHAR(255),
          quantity INT NOT NULL,
          available_quantity INT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          preview_link VARCHAR(255),
          updated_at DATETIME(6)
        )
        """)
        
        connection.execute(create_books_table_query)
        print("Ensured books table exists with the correct schema")
        
        # Close connection
        connection.close()
        print("Database initialization complete")
        return True
        
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        return False

if __name__ == "__main__":
    init_database() 