#!/usr/bin/env python3
"""
Update the existing database schema to use LONGTEXT for descriptions field.
"""

from sqlalchemy import create_engine, text

def update_schema():
    """Update the database schema to use LONGTEXT for descriptions field"""
    try:
        # Connect to the database
        engine = create_engine('mysql+pymysql://root:12345678@localhost:3306/myLib')
        connection = engine.connect()
        print("Connected to database")
        
        # Alter the descriptions column to use LONGTEXT
        alter_query = text("""
        ALTER TABLE books 
        MODIFY COLUMN descriptions LONGTEXT
        """)
        
        connection.execute(alter_query)
        print("Updated descriptions column to LONGTEXT")
        
        # Commit changes and close connection
        connection.commit()
        connection.close()
        print("Schema update complete")
        return True
        
    except Exception as e:
        print(f"Schema update error: {str(e)}")
        return False

if __name__ == "__main__":
    update_schema() 