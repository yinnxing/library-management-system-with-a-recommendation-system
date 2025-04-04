import pandas as pd
from flask import Flask, request, jsonify

# Khởi tạo Flask app
app = Flask(__name__)

# Đọc dữ liệu từ file CSV
df = pd.read_csv("/Users/Cecilia/python/BookRecommendationSystem/user_book_matrix.csv")

# Lọc và xử lý dữ liệu:
# Bỏ các cột không cần thiết (chỉ giữ lại cột User-ID và các cột sách)
df_filtered = df.dropna(axis=1, how='all')  # Loại bỏ cột không có giá trị

# Chuyển đổi dữ liệu thành ma trận người dùng - sách (User-ID làm chỉ số)
user_book_df = df_filtered.set_index('User-ID')

# Hàm gợi ý sách dựa trên tên sách
def recommend_books(book_name):
    if book_name not in user_book_df.columns:
        return {"error": "Book not found in the dataset"}

    # Tính độ tương quan giữa các cuốn sách
    corr_series = user_book_df.corrwith(user_book_df[book_name]).sort_values(ascending=False)

    # Lấy ra 5 cuốn sách có độ tương quan cao nhất
    rec_books = corr_series.head(5).index.tolist()

    return rec_books

# API endpoint nhận tên sách và trả về danh sách sách gợi ý
@app.route('/recommend', methods=['GET'])
def get_recommendations():
    book_name = request.args.get('book_name')  # Lấy tên sách từ tham số URL
    if not book_name:
        return jsonify({"error": "Book name is required"}), 400

    try:
        rec_books = recommend_books(book_name)
        return jsonify({"recommended_books": rec_books})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Chạy Flask app
if __name__ == '__main__':
    app.run(debug=True)
