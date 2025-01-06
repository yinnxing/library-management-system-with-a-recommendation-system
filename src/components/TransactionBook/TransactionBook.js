import React from 'react';
import styles from './TransactionBook.module.css'; // Đảm bảo tạo file CSS cho component này

const TransactionBook = ({ transaction }) => {
  const { book, borrowDate, dueDate, returnDate, status } = transaction;

  return (
    <div className={styles.transactionBookItem}>
      <img src={book.coverImageUrl} alt={book.title} className={styles.bookCover} />
      <div className={styles.bookDetails}>
        <h2 className={styles.bookTitle}>{book.title}</h2>
        <p className={styles.bookAuthor}>Tác giả: {book.author}</p>
        <p className={styles.bookGenre}>Thể loại: {book.genre || 'Không rõ'}</p>
        <div className={styles.transactionDetails}>
          <p><strong>Ngày mượn:</strong> {new Date(borrowDate).toLocaleDateString()}</p>
          <p><strong>Ngày trả dự kiến:</strong> {new Date(dueDate).toLocaleDateString()}</p>
          <p><strong>Ngày trả:</strong> {returnDate ? new Date(returnDate).toLocaleDateString() : 'Chưa trả'}</p>
          <p><strong>Trạng thái:</strong> {status}</p>
        </div>
      </div>
    </div>
  );
};

export default TransactionBook;
