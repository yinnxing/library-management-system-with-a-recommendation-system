import React, { useState, useEffect } from 'react';
import UserApi from '../../api/UserApi';
import { useUser } from '../../contexts/UserContext'; 
import styles from './FavoriteBookPage.module.css';
import Book from '../../components/Book/Book';


const FavoriteBookPage = () => {
  const { user } = useUser();
  const [favoriteBooks, setFavoriteBooks] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
  const fetchFavoriteBooks = async () => {
    try {
      const response = await UserApi.getFavoriteBooks(user.userId);
      console.log('Response từ API:', response);

      if (response.data.code === 0) {
        setFavoriteBooks(response.data.result);
      } else {
        console.error('Lỗi lấy danh sách yêu thích:', response.data.message);
      }
    } catch (error) {
      console.error('Lỗi gọi API:', error); 
      setLoading(false);
    }
  };

  fetchFavoriteBooks();
}, [user.userId]);


  return (
    <div className={styles.favoriteBookPage}>
      <h1 className={styles.title}>Danh sách yêu thích</h1>
      {loading ? (
        <p className={styles.loading}>Đang tải danh sách...</p>
      ) : favoriteBooks.length > 0 ? (
        <div className={styles.bookList}>
          {favoriteBooks.map((book) => (
            <Book key={book.bookId} book={book} />
          ))}
        </div>
      ) : (
        <p className={styles.noBooks}>Bạn chưa có sách yêu thích nào.</p>
      )}
    </div>
  );
};

export default FavoriteBookPage;
