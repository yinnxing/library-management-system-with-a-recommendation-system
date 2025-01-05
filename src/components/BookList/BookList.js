import React from 'react';
import styles from './BookList.module.css';
import { Link } from 'react-router-dom'; 
import { useFavorites } from '../../contexts/FavouritesContext';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import FavoriteIcon from '@mui/icons-material/Favorite';

const BookList = ({ books }) => {
    const { favoriteBooks, addToFavorites, removeFromFavorites } = useFavorites();

    const handleAddToFavorites = (bookId) => {
        if (favoriteBooks.includes(bookId)) {
            removeFromFavorites(bookId);
        } else {
            addToFavorites(bookId);
        }
    };

    return (
        <div className={styles.bookListContainer}>
            {books.map((book) => (
                <div key={book.bookId} className={styles.bookCard}>
                        <button 
                            onClick={() => handleAddToFavorites(book.bookId)} 
                            className={styles.favoriteButton}
                        >
                             {favoriteBooks.includes(book.bookId) ? (
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
                        
    
                        <Link 
                            to={`/borrow/${book.bookId}`} 
                            className={styles.borrowButton}
                            style={{ textDecoration: 'none' }}
                        >
                            <button 
                                disabled={book.availableQuantity === 0} 
                                className={styles.borrowButton}
                            >
                                Borrow Book
                            </button>
                        </Link>

                        
                    </div>
                </div>
            ))}
        </div>
    );
};

export default BookList;
