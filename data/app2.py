from sqlalchemy import create_engine, text
import random
import requests
from datetime import datetime
import pandas as pd
import time

# Tạo kết nối đến cơ sở dữ liệu (thay đổi thông tin này theo cấu hình của bạn)
# engine = create_engine('mysql+pymysql://root:12345678@localhost:3306/lib')

# Đọc file CSV
file_path = '/Users/Cecilia/python/data/title_unique.csv'
books_df = pd.read_csv(file_path)
print(books_df.head())

def fetch_book_info(title):
    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}"
    response = requests.get(url)
    data = response.json()
    
    if "items" in data:
        volume_info = data["items"][0]["volumeInfo"]
        
        # Lấy volumeId từ link hoặc từ id của item
        volume_id = data["items"][0].get("id", "")
        preview_link = volume_info.get("previewLink", "")
        
        # Tạo link nhúng
        if volume_id:
            embed_link = f"https://books.google.com.vn/books?id={volume_id}&lpg=PP1&dq=intitle:{title.replace(' ', '+')}&hl=vi&pg=PP1&output=embed"
        else:
            embed_link = "https://books.google.com.vn/books?output=embed"
        
        # Lấy ISBN
        isbn = None
        if "industryIdentifiers" in volume_info:
            for identifier in volume_info["industryIdentifiers"]:
                if identifier["type"] == "ISBN_13":
                    isbn = identifier["identifier"]
                    break
                elif identifier["type"] == "ISBN_10":
                    isbn = identifier["identifier"]
                    break
        if not isbn:
            isbn = f"ISBN{random.randint(100000, 999999)}"
        
        return {
            "title": volume_info.get("title", "Untitled Book"),
            "author": ", ".join(volume_info.get("authors", ["Unknown Author"])),
            "publisher": volume_info.get("publisher", "Default Publisher"),
            "publication_year": volume_info.get("publishedDate", "1900")[:4],
            "genre": ", ".join(volume_info.get("categories", ["No genre"])),
            "description": volume_info.get("description", "No description available."),
            "cover_image_url": volume_info.get("imageLinks", {}).get("thumbnail", "https://via.placeholder.com/150"),
            "preview_link": embed_link,
            "isbn": isbn
        }
    else:
        # Nếu không tìm thấy thông tin, trả về dữ liệu ngẫu nhiên
        return generate_random_data(title)


# Hàm phụ để tạo dữ liệu ngẫu nhiên cho các trường thiếu
def generate_random_data(title):
    return {
        "title": f"Random Book {title}",
        "author": "Unknown Author",
        "publisher": "Default Publisher",
        "publication_year": random.randint(1900, 2024),
        "genre": random.choice(["Fiction", "Non-Fiction", "Science Fiction", "Fantasy"]),
        "description": "No description available.",
        "cover_image_url": "https://via.placeholder.com/150",
        "preview_link": "",
        "isbn": f"ISBN{random.randint(100000, 999999)}"  # Tạo ISBN ngẫu nhiên
    }

# Chuẩn bị danh sách với dữ liệu được điền đầy đủ từ API nếu có
book_data_list = []

for index, row in books_df.iterrows():
    title = row.get("Book-Title", "Unknown Title")
    print(f"Đang xử lý sách: {title}")
    
    book_info = fetch_book_info(title)
    
    # Tạo dữ liệu cho một sách
    book_data = {
        "title": book_info["title"],
        "author": book_info["author"],
        "publisher": book_info["publisher"],
        "publication_year": int(book_info["publication_year"]),
        "isbn": book_info["isbn"],  # Lấy ISBN từ API
        "genre": book_info["genre"],
        "description": book_info["description"],
        "cover_image_url": book_info["cover_image_url"],
        "quantity": random.randint(0, 20),
        "available_quantity": random.randint(0, 20),
        "preview_link": book_info["preview_link"]
    }

    # Thêm dữ liệu vào danh sách
    book_data_list.append(book_data)
    
    # Đợi một thời gian ngắn để tránh bị Google Books API chặn do quá nhiều yêu cầu
    time.sleep(0.1)

# Chuyển danh sách dữ liệu thành DataFrame
books_to_export_df = pd.DataFrame(book_data_list)

# Xuất dữ liệu ra file CSV (để bạn có thể sửa thủ công)
export_file_path = '/Users/Cecilia/python/data/books_to_export.csv'
books_to_export_df.to_csv(export_file_path, index=False)

print(f"Dữ liệu đã được xuất vào file {export_file_path}. Bạn có thể chỉnh sửa thủ công và nhập lại vào cơ sở dữ liệu.")
