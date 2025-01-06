import React, {useEffect, useState} from 'react';
import styles from './BookList.module.css';
import { Link } from 'react-router-dom'; 
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import FavoriteIcon from '@mui/icons-material/Favorite';
import UserApi from '../../api/UserApi'; // Import UserApi



const BookList = ({ books, userId }) => {
  const [favoriteBooks, setFavoriteBooks] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch favorite books on initial render or when userId changes
  useEffect(() => {
    const fetchFavoriteBooks = async () => {
      try {
        const response = await UserApi.getFavoriteBooks(userId);
        if (response.data.code === 0) {
          setFavoriteBooks(response.data.result); // Assuming result contains the list of favorite books
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

  // Add a book to favorites
  const addToFavorites = async (bookId) => {
    try {
      await UserApi.addToFavorite(userId, bookId); // Call the API method from UserApi
      setFavoriteBooks((prev) => [...prev, bookId]); // Update the state to reflect the added book
    } catch (error) {
      console.error('Error adding to favorites:', error);
    }
  };

  // Remove a book from favorites
  const removeFromFavorites = async (bookId) => {
    try {
      await UserApi.removeFromFavorite(userId, bookId); 
      setFavoriteBooks((prev) => prev.filter((id) => id !== bookId)); // Update the state to reflect the removed book
    } catch (error) {
      console.error('Error removing from favorites:', error);
    }
  };

  // Handle toggle between adding/removing from favorites
  const handleFavoriteToggle = (bookId) => {
    if (favoriteBooks.includes(bookId)) {
      removeFromFavorites(bookId);
    } else {
      addToFavorites(bookId);
    }
  };

  return (
    <div className={styles.bookListContainer}>
      {loading ? (
        <p>Loading your favorite books...</p>
      ) : (
        books.map((book) => (
          <div key={book.bookId} className={styles.bookCard}>
            <button 
              onClick={() => handleFavoriteToggle(book.bookId)} 
              className={styles.favoriteButton}
            >
            {favoriteBooks.some(favBook => favBook.bookId === book.bookId) ? (
                <FavoriteIcon style={{ color: '#f0979a', fontSize: 40 }} />
                ) : (
                <FavoriteBorderIcon style={{ fontSize: 40 }} />
            )}
            </button>
            <img 
              src={book.coverImageUrl} 
              alt={book.title} 
              className={styles.bookImage}
            />
            <div className={styles.bookInfo}>
              <Link to={`/books/${book.bookId}`}>
                <h3>{book.title}</h3>
                <p><strong>Author:</strong> {book.author}</p>
                <p><strong>Available:</strong> {book.availableQuantity}</p>
              </Link>

              <Link to={`/borrow/${book.bookId}`} className={styles.borrowButton} style={{ textDecoration: 'none' }}>
                <button 
                  disabled={book.availableQuantity === 0} 
                  className={styles.borrowButton}
                >
                  Borrow Book
                </button>
              </Link>
            </div>
          </div>
        ))
      )}
    </div>
  );
};

export default BookList;
