import React from 'react';
import styles from './Book.module.css';

const Book = ({ book }) => {
  return (
    <div className={styles.bookItem}>
      <img src={book.coverImageUrl} alt={book.title} className={styles.bookCover} />
      <div className={styles.bookDetails}>
        <h2 className={styles.bookTitle}>{book.title}</h2>
        <p className={styles.bookAuthor}>Tác giả: {book.author}</p>
        <p className={styles.bookGenre}>Thể loại: {book.genre || 'Không rõ'}</p>
        <p className={styles.bookDescription}>{book.descriptions}</p>
        <a
          href={book.previewLink}
          target="_blank"
          rel="noopener noreferrer"
          className={styles.previewLink}
        >
          Xem chi tiết
        </a>
      </div>
    </div>
  );
};

export default Book;
