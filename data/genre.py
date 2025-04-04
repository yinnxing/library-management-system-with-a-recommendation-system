import requests

# Tìm kiếm sách với ISBN
isbn = "3789119520"  # Thay thế bằng ISBN bạn muốn tìm
url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"

response = requests.get(url)
book_data = response.json()

# Kiểm tra xem sách có tồn tại không
if "items" in book_data:
    book = book_data["items"][0]
    title = book["volumeInfo"].get("title", "No title")
    authors = ", ".join(book["volumeInfo"].get("authors", ["Unknown author"]))
    categories = ", ".join(book["volumeInfo"].get("categories", ["No category"]))
    preview_link = book["volumeInfo"].get("previewLink", "No preview link")

    print(f"Title: {title}")
    print(f"Authors: {authors}")
    print(f"Categories: {categories}")
    print(f"Preview: {preview_link}")
else:
    print("No book found with this ISBN.")
