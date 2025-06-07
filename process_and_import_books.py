#!/usr/bin/env python3
"""
Main script to export preprocessed books from the recommender system
and import them into the MySQL database with Google Books API data.
"""

import os
import sys
import traceback
from export_preprocessed_books import export_preprocessed_books
from import_books_into_database import import_books_to_database
from init_database import init_database

def main():
    try:
        print("=" * 50)
        print("STEP 1: Initializing database")
        print("=" * 50)
        
        # Initialize database
        if not init_database():
            print("Database initialization failed. Exiting.")
            return 1
        
        print("\n" + "=" * 50)
        print("STEP 2: Exporting preprocessed books from recommender system")
        print("=" * 50)
        
        # Export books to CSV
        try:
            csv_path = export_preprocessed_books()
        except Exception as e:
            print(f"\nError during book export: {str(e)}")
            traceback.print_exc()
            return 1
        
        if csv_path and os.path.exists(csv_path):
            print("\n" + "=" * 50)
            print(f"STEP 3: Importing books from {csv_path} into database")
            print("=" * 50)
            
            # Import books to database
            try:
                import_books_to_database()
                print("\nProcess completed successfully!")
                return 0
            except Exception as e:
                print(f"\nError during book import: {str(e)}")
                traceback.print_exc()
                return 1
        else:
            print("\nError: Failed to export books. Database import skipped.")
            return 1
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 