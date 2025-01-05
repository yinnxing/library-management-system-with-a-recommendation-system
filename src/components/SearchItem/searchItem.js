import React from 'react';
import styles from './searchItem.module.css';
import { Link } from 'react-router-dom'; 
const SearchItem = ({ book, onSelect }) => {
    return (
        <div className={styles.resultItem}>
            <div className={styles.itemContent}>
                    <img 
                        src={book.coverImageUrl} 
                        alt={book.title} 
                        className={styles.itemImage} 
                    />
                    <Link to={`/books/${book.bookId}`}>
                        <div className={styles.itemInfo}>
                            <h4>{book.title}</h4>
                            <p>Available: {book.availableQuantity}</p>
                        </div>
                    </Link>
            </div>
        </div>
    );
};



export default SearchItem;
