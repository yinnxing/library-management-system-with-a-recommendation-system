import React from 'react';
import styles from './TransactionBook.module.css';
import '../../styles/design-system.css';

const TransactionBook = ({ transaction }) => {
  const { transactionId, book, borrowDate, dueDate, returnDate, pickupDeadline, overdueFee, status } = transaction;

  // Format date helper function
  const formatDate = (dateString) => {
    if (!dateString) return 'Không có';
    return new Date(dateString).toLocaleDateString('vi-VN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Format currency helper function
  const formatCurrency = (amount) => {
    if (amount === null || amount === undefined) return 'Không có';
    return new Intl.NumberFormat('vi-VN', {
      style: 'currency',
      currency: 'VND'
    }).format(amount);
  };

  // Get status styling
  const getStatusClass = (status) => {
    switch (status?.toLowerCase()) {
      case 'pending':
        return styles.statusPending;
      case 'borrowed':
        return styles.statusBorrowed;
      case 'returned':
        return styles.statusReturned;
      case 'overdue':
        return styles.statusOverdue;
      case 'cancelled':
        return styles.statusCancelled;
      default:
        return styles.statusDefault;
    }
  };

  // Check if overdue
  const isOverdue = dueDate && new Date(dueDate) < new Date() && !returnDate;

  return (
    <div className={styles.transactionBookItem}>
      <div className={styles.bookCoverContainer}>
        <img src={book.coverImageUrl} alt={book.title} className={styles.bookCover} />
      </div>
      
      <div className={styles.bookDetails}>
        <div className={styles.bookInfo}>
          <h2 className={styles.bookTitle}>{book.title}</h2>
          <p className={styles.bookAuthor}>Tác giả: {book.author}</p>
          <p className={styles.bookMeta}>
            <span className={styles.genre}>{book.genre || 'Không rõ'}</span>
            <span className={styles.separator}>•</span>
            <span className={styles.year}>{book.publicationYear}</span>
          </p>
        </div>

        <div className={styles.transactionInfo}>
          <div className={styles.transactionHeader}>
            <h3 className={styles.transactionTitle}>Thông tin giao dịch</h3>
            <span className={`${styles.status} ${getStatusClass(status)}`}>
              {status === 'PENDING' ? 'Chờ nhận sách' : 
               status === 'BORROWED' ? 'Đang mượn' : 
               status === 'RETURNED' ? 'Đã trả' : 
               status === 'OVERDUE' ? 'Quá hạn' : 
               status === 'CANCELLED' ? 'Đã hủy' : status}
            </span>
          </div>

          <div className={styles.transactionDetails}>
            <div className={styles.detailRow}>
              <span className={styles.label}>Mã giao dịch:</span>
              <span className={styles.value}>{transactionId}</span>
            </div>

            {/* Display fields based on transaction status */}
            {status === 'PENDING' && (
              <>
                <div className={styles.detailRow}>
                  <span className={styles.label}>Ngày đặt trước:</span>
                  <span className={styles.value}>{formatDate(borrowDate)}</span>
                </div>
                {pickupDeadline && (
                  <div className={styles.detailRow}>
                    <span className={styles.label}>Hạn nhận sách:</span>
                    <span className={styles.value}>{formatDate(pickupDeadline)}</span>
                  </div>
                )}
                {dueDate && (
                  <div className={styles.detailRow}>
                    <span className={styles.label}>Hạn trả dự kiến:</span>
                    <span className={styles.value}>{formatDate(dueDate)}</span>
                  </div>
                )}
              </>
            )}

            {status === 'BORROWED' && (
              <>
                <div className={styles.detailRow}>
                  <span className={styles.label}>Ngày mượn:</span>
                  <span className={styles.value}>{formatDate(borrowDate)}</span>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.label}>Hạn trả:</span>
                  <span className={`${styles.value} ${isOverdue ? styles.overdue : ''}`}>
                    {formatDate(dueDate)}
                    {isOverdue && <span className={styles.overdueLabel}>Quá hạn</span>}
                  </span>
                </div>
              </>
            )}

            {status === 'RETURNED' && (
              <>
                <div className={styles.detailRow}>
                  <span className={styles.label}>Ngày mượn:</span>
                  <span className={styles.value}>{formatDate(borrowDate)}</span>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.label}>Hạn trả:</span>
                  <span className={styles.value}>{formatDate(dueDate)}</span>
                </div>
                {returnDate && (
                  <div className={styles.detailRow}>
                    <span className={styles.label}>Ngày trả thực tế:</span>
                    <span className={styles.value}>{formatDate(returnDate)}</span>
                  </div>
                )}
              </>
            )}

            {status === 'OVERDUE' && (
              <>
                <div className={styles.detailRow}>
                  <span className={styles.label}>Ngày mượn:</span>
                  <span className={styles.value}>{formatDate(borrowDate)}</span>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.label}>Hạn trả:</span>
                  <span className={`${styles.value} ${styles.overdue}`}>
                    {formatDate(dueDate)}
                    <span className={styles.overdueLabel}>Đã quá hạn</span>
                  </span>
                </div>
                {overdueFee !== null && overdueFee !== undefined && overdueFee > 0 && (
                  <div className={styles.detailRow}>
                    <span className={styles.label}>Phí phạt quá hạn:</span>
                    <span className={`${styles.value} ${styles.feeAmount}`}>
                      {formatCurrency(overdueFee)}
                    </span>
                  </div>
                )}
              </>
            )}

            {status === 'CANCELLED' && (
              <>
                <div className={styles.detailRow}>
                  <span className={styles.label}>Ngày đặt trước:</span>
                  <span className={styles.value}>{formatDate(borrowDate)}</span>
                </div>
                {pickupDeadline && (
                  <div className={styles.detailRow}>
                    <span className={styles.label}>Hạn nhận sách:</span>
                    <span className={styles.value}>{formatDate(pickupDeadline)}</span>
                  </div>
                )}
                <div className={styles.detailRow}>
                  <span className={styles.label}>Lý do hủy:</span>
                  <span className={styles.value}>Quá hạn nhận sách</span>
                </div>
              </>
            )}

            {/* Fallback for unknown status */}
            {!['PENDING', 'BORROWED', 'RETURNED', 'OVERDUE', 'CANCELLED'].includes(status) && (
              <>
                <div className={styles.detailRow}>
                  <span className={styles.label}>Ngày mượn:</span>
                  <span className={styles.value}>{formatDate(borrowDate)}</span>
                </div>
                {dueDate && (
                  <div className={styles.detailRow}>
                    <span className={styles.label}>Hạn trả:</span>
                    <span className={styles.value}>{formatDate(dueDate)}</span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TransactionBook;
