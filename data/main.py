import requests

# Tìm kiếm sách trên Google Books
query = "Python programming"
url = f"https://www.googleapis.com/books/v1/volumes?q={query}"

response = requests.get(url)
books = response.json()

for book in books.get("items", []):
    title = book["volumeInfo"].get("title", "No title")
    preview_link = book["volumeInfo"].get("previewLink", "No preview link")
    print(f"Title: {title}\nPreview: {preview_link}\n")
