import pandas as pd

# Đọc dữ liệu từ các tệp CSV
book = pd.read_csv('./Books.csv', low_memory=False)
rating = pd.read_csv('./Ratings.csv', low_memory=False)
users = pd.read_csv('./Users.csv', low_memory=False)

# Kết hợp các bảng dữ liệu
df1 = book.merge(rating, how="left", on="ISBN")
df_ = df1.merge(users, how="left", on="User-ID")

# Tạo một bản sao của dữ liệu đã kết hợp
df = df_.copy()

# Xem trước dữ liệu
print(df.head())
