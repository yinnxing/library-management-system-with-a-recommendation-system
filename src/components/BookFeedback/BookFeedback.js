import React, {useEffect, useState} from 'react';
import '../../styles/design-system.css';
import styles from './BookFeedback.module.css';
import { Link } from 'react-router-dom'; 
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import FavoriteIcon from '@mui/icons-material/Favorite';
import UserApi from '../../api/UserApi'; 
import { Rate } from "antd"; 

const BookFeedback = ({ books, userId }) => {
  const [favoriteBooks, setFavoriteBooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [ratings, setRatings] = useState({});
  const [feedbackSubmitted, setFeedbackSubmitted] = useState({});

  useEffect(() => {
    if (userId) {
      fetchFavoriteBooks();
    } else {
      setLoading(false);
    }
  }, [userId]);

    const fetchFavoriteBooks = async () => {
      try {
        const response = await UserApi.getFavoriteBooks(userId);
        if (response.data.code === 0) {
          const bookIds = response.data.result.map((book) => book.bookId);
          setFavoriteBooks(bookIds);
        } else {
          console.error("Error fetching favorite books:", response.data.message);
        }
      } catch (error) {
        console.error("Error calling API:", error);
      } finally {
        setLoading(false);
      }
    };

  const addToFavorites = async (bookId) => {
    try {
      await UserApi.addToFavorite(userId, bookId);
    } catch (error) {
      console.error("Error adding to favorites:", error);
    }
  };

  const removeFromFavorites = async (bookId) => {
    try {
      await UserApi.removeFromFavorite(userId, bookId);
    } catch (error) {
      console.error("Error removing from favorites:", error);
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
      console.error("Error toggling favorite:", error);
    }
  };

 const handleRating = async (bookId, rating) => {
  try {
      setRatings((prev) => ({ ...prev, [bookId]: rating }));
    const feedbackData = {
      userId,
      bookId,
      rating,
    };
      const response = await UserApi.submitFeedback(feedbackData);
    if (response.data.code === 0) {
        setFeedbackSubmitted((prev) => ({ ...prev, [bookId]: true }));
        // Show success message briefly
        setTimeout(() => {
          setFeedbackSubmitted((prev) => ({ ...prev, [bookId]: false }));
        }, 2000);
    } else {
      console.error("Error submitting feedback:", response.data.message);
    }
  } catch (error) {
    console.error("Error submitting feedback:", error);
  }
};

  if (loading) {
    return (
      <div className={styles.bookListContainer}>
        <div className={styles.loadingCard}>
          <div className={styles.loadingSpinner}></div>
        </div>
      </div>
    );
  }

  if (!books || books.length === 0) {
    return (
      <div className={styles.bookListContainer}>
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>📚</div>
          <h3>Chưa có gợi ý sách</h3>
          <p>Hiện tại chưa có sách nào được gợi ý cho bạn. Hãy thử đánh giá một số sách để nhận được gợi ý tốt hơn!</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.bookListContainer}>
      {books.map((book) => {
          const isFavorite = favoriteBooks.includes(book.bookId);
        const isRated = ratings[book.bookId] > 0;
        const showFeedbackSuccess = feedbackSubmitted[book.bookId];

          return (
            <div key={book.bookId} className={styles.bookCard}>
            {/* Favorite Button */}
              <button
                onClick={() => handleFavoriteToggle(book.bookId)}
                className={styles.favoriteButton}
              title={isFavorite ? "Xóa khỏi yêu thích" : "Thêm vào yêu thích"}
              >
                {isFavorite ? (
                <FavoriteIcon style={{ color: "#f0979a", fontSize: 24 }} />
                ) : (
                <FavoriteBorderIcon style={{ fontSize: 24 }} />
                )}
              </button>

            {/* Book Image */}
              <img
                src={book.coverImageUrl}
                alt={book.title}
                className={styles.bookImage}
              />

            {/* Book Info */}
              <div className={styles.bookInfo}>
                <Link to={`/books/${book.bookId}`}>
                  <h3>{book.title}</h3>
                <p><strong>Tác giả:</strong> {book.author}</p>
                <p><strong>Còn lại:</strong> {book.availableQuantity || 'N/A'}</p>
                </Link>

              {/* Rating Section */}
              <div className={styles.ratingSection}>
                <div className={styles.ratingLabel}>
                  <span className={styles.ratingIcon}>⭐</span>
                  Đánh giá gợi ý:
                </div>
                <Rate
                  onChange={(value) => handleRating(book.bookId, value)}
                  value={ratings[book.bookId] || 0}
                  className={styles.customRate}
                  size="small"
                />
                {showFeedbackSuccess && (
                  <div className={styles.feedbackSuccess}>
                    ✅ Cảm ơn đánh giá!
                  </div>
                )}
              </div>

              {/* Borrow Button */}
                <Link
                  to={`/borrow/${book.bookId}`}
                  className={styles.borrowButton}
                style={{ textDecoration: 'none' }}
                >
                  <button
                    disabled={book.availableQuantity === 0}
                    className={styles.borrowButton}
                  >
                  {book.availableQuantity > 0 ? 'Mượn sách' : 'Hết sách'}
                  </button>
                </Link>
              </div>
            </div>
          );
      })}
    </div>
  );
};

export default BookFeedback;

