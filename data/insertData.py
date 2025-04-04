import pandas as pd
import random
from sqlalchemy import create_engine, text

# Đường dẫn file CSV
file_path = '/Users/Cecilia/python/data/books_to_export.csv'

# Đọc file CSV
books_to_import_df = pd.read_csv(file_path)

# Thay thế giá trị NaN bằng giá trị mặc định
books_to_import_df.fillna({
    "preview_link": ""  # Thay thế NaN trong cột preview_link bằng chuỗi rỗng
}, inplace=True)

# Kết nối đến cơ sở dữ liệu
engine = create_engine('mysql+pymysql://root:12345678@localhost:3306/lib2')

# Lặp qua từng dòng để chèn dữ liệu
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
        "quantity": row["quantity"],
        "available_quantity": row["available_quantity"],
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
