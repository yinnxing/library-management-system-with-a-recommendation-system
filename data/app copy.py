from sqlalchemy import create_engine, text
import random
import requests
from datetime import datetime
import pandas as pd
import time

# Tạo kết nối đến cơ sở dữ liệu (thay đổi thông tin này theo cấu hình của bạn)
engine = create_engine('mysql+pymysql://root:12345678@localhost:3306/myLib')
# Đọc file CSV đã chỉnh sửa
books_to_import_df = pd.read_csv('/Users/Cecilia/python/data/books_to_export2.csv')

# Chèn lại dữ liệu vào cơ sở dữ liệ
for index, row in books_to_import_df.iterrows():
    book_data = {
        "title": row["title"],
        "author": row["author"],
        "publisher": row["publisher"],
        "publication_year": int(row["publication_year"]),
        "isbn": row["isbn"],
        "genre": row["genre"],
        "descriptions": row["description"],
        "cover_image_url": row["cover_image_url"],
        "quantity": random.randint(0, 20),
        "available_quantity": random.randint(0, 20),
        "preview_link": row["preview_link"]
    }

    try:
        with engine.begin() as connection:
            query = text("""
                INSERT INTO Books (title, author, publisher, publication_year, isbn, genre, descriptions, cover_image_url, quantity, available_quantity, preview_link)
                VALUES (:title, :author, :publisher, :publication_year, :isbn, :genre, :descriptions, :cover_image_url, :quantity, :available_quantity, :preview_link)
            """)
            connection.execute(query, book_data)
            print(f"Đã chèn thành công ISBN {book_data['isbn']} vào cơ sở dữ liệu.")
    except Exception as e:
        print(f"Lỗi khi chèn ISBN {book_data['isbn']}: {e}")
