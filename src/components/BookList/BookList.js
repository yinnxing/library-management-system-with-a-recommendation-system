import React, {useEffect, useState} from 'react';
import styles from './BookList.module.css';
import { Link } from 'react-router-dom'; 
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import FavoriteIcon from '@mui/icons-material/Favorite';
import UserApi from '../../api/UserApi'; 



const BookList = ({ books, userId }) => {
  const [favoriteBooks, setFavoriteBooks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
  const fetchFavoriteBooks = async () => {
    try {
      const response = await UserApi.getFavoriteBooks(userId);
      if (response.data.code === 0) {
        const bookIds = response.data.result.map(book => book.bookId);
        setFavoriteBooks(bookIds);
      } else {
        console.error('Error fetching favorite books:', response.data.message);
      }
    } catch (error) {
      console.error('Error calling API:', error);
    } finally {
      setLoading(false);
    }
  };

  fetchFavoriteBooks();
}, [userId]);


  const addToFavorites = async (bookId) => {
    try {
      await UserApi.addToFavorite(userId, bookId); 
    } catch (error) {
      console.error('Error adding to favorites:', error);
    }
  };

  const removeFromFavorites = async (bookId) => {
    try {
      await UserApi.removeFromFavorite(userId, bookId); 
    } catch (error) {
      console.error('Error removing from favorites:', error);
    }
  };

 const handleFavoriteToggle = async (bookId) => {
  try {
    if (favoriteBooks.includes(bookId)) {
      await removeFromFavorites(bookId); 
      setFavoriteBooks((prev) => prev.filter((id) => id !== bookId));
    } else {
      await addToFavorites(bookId); 
      setFavoriteBooks((prev) => [...prev, bookId]);
    }
  } catch (error) {
    console.error('Error toggling favorite:', error);
  }
};

return (
  <div className={styles.bookListContainer}>
    {loading ? (
      <p>Loading your favorite books...</p>
    ) : (
      books.map((book) => {
        const isFavorite = favoriteBooks.includes(book.bookId);

        return (
          <div key={book.bookId} className={styles.bookCard}>
            {/* Nút yêu thích */}
            <button 
              onClick={() => handleFavoriteToggle(book.bookId)} 
              className={styles.favoriteButton}
            >
              {isFavorite ? (
                <FavoriteIcon style={{ color: '#f0979a', fontSize: 40 }} />
              ) : (
                <FavoriteBorderIcon style={{ fontSize: 40 }} />
              )}
            </button>

            {/* Ảnh bìa sách */}
            <img 
              src={book.coverImageUrl} 
              alt={book.title} 
              className={styles.bookImage}
            />

            {/* Thông tin sách */}
            <div className={styles.bookInfo}>
              <Link to={`/books/${book.bookId}`}>
                <h3>{book.title}</h3>
                <p><strong>Author:</strong> {book.author}</p>
                <p><strong>Available:</strong> {book.availableQuantity}</p>
              </Link>

              {/* Nút mượn sách */}
              <Link 
                to={`/borrow/${book.bookId}`} 
                className={styles.borrowButton} 
                style={{ textDecoration: 'none' }}
              >
                <button 
                  disabled={book.availableQuantity === 0} 
                  className={styles.borrowButton}
                >
                  {book.availableQuantity > 0 ? 'Borrow Book' : 'Unavailable'}
                </button>
              </Link>
            </div>
          </div>
        );
      })
    )}
  </div>
);



};

export default BookList;

