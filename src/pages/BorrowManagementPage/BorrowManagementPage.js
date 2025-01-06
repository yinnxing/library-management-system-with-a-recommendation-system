import React, { useState, useEffect} from 'react';
import styles from './BorrowManagementPage.module.css';
import BookIcon from '@mui/icons-material/Book';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import UserApi from '../../api/UserApi';
import { useUser } from '../../contexts/UserContext'; 
import Book from '../../components/TransactionBook/TransactionBook'; 

const BorrowManagementPage = () => {
  const [selectedTab, setSelectedTab] = useState('processing');
  const { user } = useUser();
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);

  // Hàm gọi API tương ứng theo trạng thái
  const fetchBooksByTab = async () => {
    setLoading(true);
    try {
      let response;
      if (selectedTab === 'processing') {
        response = await UserApi.getPendingTransactions(user.userId); // Sách đang xử lý
      } else if (selectedTab === 'borrowing') {
        response = await UserApi.getBorrowedBooks(user.userId); // Lấy sách đã mượn (thay đổi API)
      } else if (selectedTab === 'returned') {
        response = await UserApi.getReturnTransactions(user.userId); // Sách đã trả
      }

      // Chỉnh sửa dữ liệu để truyền vào TransactionBook
      const fetchedBooks = response.data.map((transaction) => ({
        transactionId: transaction.transactionId,
        book: transaction.book,
        borrowDate: transaction.borrowDate,
        dueDate: transaction.dueDate,
        returnDate: transaction.returnDate,
        status: transaction.status,
      }));
      setBooks(fetchedBooks);
    } catch (error) {
      console.error('Error fetching books:', error);
      setBooks([]);
    } finally {
      setLoading(false);
    }
  };

  // Gọi API khi thay đổi tab
  useEffect(() => {
    fetchBooksByTab();
  }, [selectedTab, user.userId]);

  return (
    <div className={styles.borrowManagementPage}>
      {/* Tabs */}
      <div className={styles.tabs}>
        {[
          { label: 'Sách đang xử lý', value: 'processing', Icon: HourglassEmptyIcon },
          { label: 'Sách đã mượn', value: 'borrowing', Icon: BookIcon },
          { label: 'Sách đã trả', value: 'returned', Icon: CheckCircleIcon },
        ].map(({ label, value, Icon }) => (
          <button
            key={value}
            className={`${styles.tabButton} ${selectedTab === value ? styles.activeTab : ''}`}
            onClick={() => setSelectedTab(value)}
          >
            <Icon className={styles.icon} />
            {label}
          </button>
        ))}
      </div>

      {/* Nội dung */}
      <div className={styles.tabContent}>
        {loading ? (
          <p className={styles.loading}>Đang tải dữ liệu...</p>
        ) : books.length > 0 ? (
          <ul className={styles.bookList}>
            {books.map((transaction) => (
              <li key={transaction.transactionId} className={styles.bookItem}>
                <Book transaction={transaction} /> 
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
