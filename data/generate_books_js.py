import pandas as pd
import requests
from datetime import datetime
import json
import random
import time

# Đọc file CSV
file_path = 'common_book_isbn.csv'  # file chỉ chứa isbn của sách 
books_df = pd.read_csv(file_path)

# Danh sách các thể loại ngẫu nhiên (nếu không có dữ liệu thể loại từ API)
genres = ["Fiction", "Non-Fiction", "Science Fiction", "Fantasy", "Biography", "Mystery", "Romance", "Thriller", "Historical", "Self-Help"]

# Hàm gọi Google Books API để lấy thông tin sách qua ISBN
def fetch_book_info(isbn):
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    response = requests.get(url)
    data = response.json()
    
    if "items" in data:
        volume_info = data["items"][0]["volumeInfo"]
        return {
            "author": ", ".join(volume_info.get("authors", ["Unknown Author"])),
            "publisher": volume_info.get("publisher", "Default Publisher"),
            "publication_year": volume_info.get("publishedDate", "1900")[:4],  # Lấy năm
            "genre": ", ".join(volume_info.get("categories", ["No genre"])),
            "description": volume_info.get("description", "No description available."),
            "cover_image_url": volume_info.get("imageLinks", {}).get("thumbnail", "https://via.placeholder.com/150")
        }
    else:
        # Nếu không tìm thấy thông tin, trả về dữ liệu ngẫu nhiên
        return generate_random_data()

# Hàm phụ để tạo dữ liệu ngẫu nhiên cho các trường thiếu
def generate_random_data():
    return {
        "author": "Unknown Author",
        "publisher": "Default Publisher",
        "publication_year": random.randint(1900, 2024),
        "genre": random.choice(genres),
        "description": "No description available.",
        "cover_image_url": "https://via.placeholder.com/150"
    }

# Chuẩn bị danh sách sách với dữ liệu được điền đầy đủ từ API nếu có
books = []
for index, row in books_df.iterrows():
    isbn = row.get("ISBN", f"ISBN{random.randint(100000, 999999)}")
    book_info = fetch_book_info(isbn)
    
    book = {
        "book_id": index + 1,
        "title": row.get("Book-Title", "Untitled Book") or "Untitled Book",
        "author": book_info["author"],
        "publisher": book_info["publisher"],
        "publication_year": int(book_info["publication_year"]),
        "isbn": isbn,
        "genre": book_info["genre"],
        "description": book_info["description"],
        "cover_image_url": book_info["cover_image_url"],
        "quantity": int(row.get("quantity", random.randint(1, 50))),
        "available_quantity": int(row.get("available_quantity", random.randint(1, 50))),
        "created_at": datetime.now().isoformat()
    }
    books.append(book)
    
    # Đợi một thời gian ngắn để tránh bị Google Books API chặn do quá nhiều yêu cầu
    time.sleep(0.1)

# Xuất dữ liệu ra file JavaScript với định dạng dễ đọc
books_js_content = "const books = " + json.dumps(books, indent=2) + ";\n\nexport default books;"

# Lưu vào file books.js
output_file_path = 'allbooks.js'
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(books_js_content)

print(f"File books.js đã được tạo tại {output_file_path}")
