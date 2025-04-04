from sqlalchemy import create_engine, text
import random
import requests
from datetime import datetime
import pandas as pd
import time

# Tạo kết nối đến cơ sở dữ liệu (thay đổi thông tin này theo cấu hình của bạn)
engine = create_engine('mysql+pymysql://root:12345678@localhost:3306/lib')

# Đọc file CSV
file_path = '/Users/Cecilia/python/data/common_book_isbn.csv'
books_df = pd.read_csv(file_path)

# Hàm gọi Google Books API để lấy thông tin sách qua ISBN
def fetch_book_info(isbn):
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    response = requests.get(url)
    data = response.json()
    
    if "items" in data:
        volume_info = data["items"][0]["volumeInfo"]
        return {
            "title": volume_info.get("title", "Untitled Book"),
            "author": ", ".join(volume_info.get("authors", ["Unknown Author"])),
            "publisher": volume_info.get("publisher", "Default Publisher"),
            "publication_year": volume_info.get("publishedDate", "1900")[:4],
            "genre": ", ".join(volume_info.get("categories", ["No genre"])),
            "description": volume_info.get("description", "No description available."),
            "cover_image_url": volume_info.get("imageLinks", {}).get("thumbnail", "https://via.placeholder.com/150"),
            "preview_link": volume_info.get("previewLink", "")
        }
    else:
        # Nếu không tìm thấy thông tin, trả về dữ liệu ngẫu nhiên
        return generate_random_data(isbn)

# Hàm phụ để tạo dữ liệu ngẫu nhiên cho các trường thiếu
def generate_random_data(isbn):
    return {
        "title": f"Random Book {isbn}",
        "author": "Unknown Author",
        "publisher": "Default Publisher",
        "publication_year": random.randint(1900, 2024),
        "genre": random.choice(["Fiction", "Non-Fiction", "Science Fiction", "Fantasy"]),
        "description": "No description available.",
        "cover_image_url": "https://via.placeholder.com/150",
        "preview_link": ""
    }

# Chuẩn bị danh sách sách với dữ liệu được điền đầy đủ từ API nếu có
for index, row in books_df.iterrows():
    isbn = row.get("ISBN", f"ISBN{random.randint(100000, 999999)}")
    print(f"Đang xử lý ISBN: {isbn}")
    
    book_info = fetch_book_info(isbn)
    
    # Tạo dữ liệu cho một sách
    book_data = {
        "title": book_info["title"],
        "author": book_info["author"],
        "publisher": book_info["publisher"],
        "publication_year": int(book_info["publication_year"]),
        "isbn": isbn,
        "genre": book_info["genre"],
        "description": book_info["description"],
        "cover_image_url": book_info["cover_image_url"],
        "quantity": random.randint(0, 20),
        "available_quantity": random.randint(0, 20),
        "preview_link": book_info["preview_link"]
    }

    try:
        with engine.begin() as connection:
            # Sử dụng text() để định nghĩa câu lệnh SQL
            query = text("""
                INSERT INTO Books (title, author, publisher, publication_year, isbn, genre, description, cover_image_url, quantity, available_quantity, preview_link)
                VALUES (:title, :author, :publisher, :publication_year, :isbn, :genre, :description, :cover_image_url, :quantity, :available_quantity, :preview_link)
            """)
            
            # Thay vì sử dụng `**book_data`, ta sẽ dùng params
            connection.execute(query, {
                "title": book_data["title"],
                "author": book_data["author"],
                "publisher": book_data["publisher"],
                "publication_year": book_data["publication_year"],
                "isbn": book_data["isbn"],
                "genre": book_data["genre"],
                "description": book_data["description"],
                "cover_image_url": book_data["cover_image_url"],
                "quantity": book_data["quantity"],
                "available_quantity": book_data["available_quantity"],
                "preview_link": book_data["preview_link"]
            })
            print(f"Đã chèn thành công ISBN {isbn} vào cơ sở dữ liệu")
    except Exception as e:
        print(f"Lỗi khi chèn ISBN {isbn}: {e}")
    
    # Đợi một thời gian ngắn để tránh bị Google Books API chặn do quá nhiều yêu cầu
    time.sleep(0.1)
    