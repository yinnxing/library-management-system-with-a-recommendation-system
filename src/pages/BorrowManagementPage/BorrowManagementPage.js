import React, { useState } from 'react';
import styles from './BorrowManagementPage.module.css';
import BookIcon from '@mui/icons-material/Book';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const BorrowManagementPage = () => {
  const [selectedTab, setSelectedTab] = useState('processing');
  const [books, setBooks] = useState([
    {
      id: 1,
      title: 'Sách A',
      author: 'Tác giả 1',
      requestDate: '2024-11-01',
      dueDate: '2024-11-15',
      returnDate: '2024-11-20',
    },
    {
      id: 2,
      title: 'Sách B',
      author: 'Tác giả 2',
      requestDate: '2024-10-25',
      dueDate: '2024-11-10',
      returnDate: '2024-11-12',
    },
  ]); 
  const [loading, setLoading] = useState(false);

  return (
    <div className={styles.borrowManagementPage}>
      <div className={styles.tabs}>
        <button
          className={`${styles.tabButton} ${
            selectedTab === 'processing' ? styles.activeTab : ''
          }`}
          onClick={() => setSelectedTab('processing')}
        >
          <HourglassEmptyIcon className={styles.icon} />
          Sách đang xử lý
        </button>
        <button
          className={`${styles.tabButton} ${
            selectedTab === 'borrowing' ? styles.activeTab : ''
          }`}
          onClick={() => setSelectedTab('borrowing')}
        >
          <BookIcon className={styles.icon} />
          Sách đang mượn
        </button>
        <button
          className={`${styles.tabButton} ${
            selectedTab === 'returned' ? styles.activeTab : ''
          }`}
          onClick={() => setSelectedTab('returned')}
        >
          <CheckCircleIcon className={styles.icon} />
          Sách đã trả
        </button>
      </div>
      <div className={styles.tabContent}>
        {loading ? (
          <p className={styles.loading}>Đang tải dữ liệu...</p>
        ) : books.length > 0 ? (
          <ul className={styles.bookList}>
            {books.map((book) => (
              <li key={book.id} className={styles.bookItem}>
                <span className={styles.bookTitle}>{book.title}</span> - {book.author} <br />
                {selectedTab === 'processing' && (
                  <span className={styles.bookDetails}>Ngày yêu cầu: {book.requestDate}</span>
                )}
                {selectedTab === 'borrowing' && (
                  <span className={styles.bookDetails}>Hạn trả: {book.dueDate}</span>
                )}
                {selectedTab === 'returned' && (
                  <span className={styles.bookDetails}>Ngày trả: {book.returnDate}</span>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p>Không có sách nào trong danh mục này.</p>
        )}
      </div>
    </div>
  );
};

export default BorrowManagementPage;
