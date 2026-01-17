import React, { useState, useEffect } from 'react';
import AdminApi from '../../api/AdminApi';
import styles from './TransactionManagement.module.css'; 
const TransactionManagement = () => {
  const [transactions, setTransactions] = useState([]);
  const [allTransactions, setAllTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('PENDING');
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [totalElements, setTotalElements] = useState(0);
  
  // Modal state for overdue fee confirmation
  const [showOverdueModal, setShowOverdueModal] = useState(false);
  const [selectedTransaction, setSelectedTransaction] = useState(null);

  const transactionsPerPage = 10;

  const statusConfig = {
    PENDING: {
      label: 'Chờ mượn sách',
      icon: '⏳',
      color: 'warning',
      description: 'Chưa nhận sách tại thư viện',
      action: 'Xác nhận mượn'
    },
    BORROWED: {
      label: 'Đã mượn sách',
      icon: '📖',
      color: 'info',
      description: 'Sách đã được nhận tại thư viện',
      action: 'Xác nhận trả'
    },
    RETURNED: {
      label: 'Đã trả sách',
      icon: '✅',
      color: 'success',
      description: 'Sách đã được trả lại',
      action: null
    },
    CANCELLED: {
      label: 'Đã hủy',
      icon: '❌',
      color: 'error',
      description: 'Giao dịch bị hủy',
      action: null
    },
    OVERDUE: {
      label: 'Quá hạn',
      icon: '⚠️',
      color: 'error',
      description: 'Quá hạn mượn sách',
      action: 'Xác nhận trả'
    }
  };

  useEffect(() => {
    fetchAllTransactions();
  }, []);

  useEffect(() => {
    fetchTransactions();
  }, [statusFilter, currentPage]);

  useEffect(() => {
    if (currentPage !== 1) {
      setCurrentPage(1);
    } else {
      fetchTransactions();
    }
  }, [searchTerm]);



  const fetchAllTransactions = async () => {
    try {
      const response = await AdminApi.getTransactions({}, { page: 1, size: 1000 });
      if (response.data.code === 0) {
        setAllTransactions(response.data.result.content);
      }
    } catch (error) {
      console.error('Error fetching all transactions:', error);
    }
  };

  const fetchTransactions = async () => {
    setLoading(true);
    try {
      const response = await AdminApi.getTransactions(
        { status: statusFilter },
        { page: currentPage, size: transactionsPerPage }
      );
      
      if (response.data.code === 0) {
        const result = response.data.result;
        let filteredTransactions = result.content;
        
        // Apply search filter
        if (searchTerm) {
          filteredTransactions = result.content.filter(transaction =>
            transaction.book.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
            transaction.transactionId.toString().includes(searchTerm) ||
            transaction.username?.toLowerCase().includes(searchTerm.toLowerCase())
          );
        }
        
        setTransactions(filteredTransactions);
        setTotalPages(result.totalPages);
        setTotalElements(result.totalElements);
        
        // Debug: Log overdue transactions to check fee data
        if (statusFilter === 'OVERDUE') {
          console.log('Overdue transactions:', filteredTransactions);
          filteredTransactions.forEach(t => {
            console.log(`Transaction ${t.transactionId}:`, {
              overdueFee: t.overdueFee,
              dueDate: t.dueDate,
              status: t.status,
              calculatedFee: calculateOverdueFee(t),
              overdueDays: getOverdueDays(t)
            });
          });
        }
      }
    } catch (error) {
      console.error('Error fetching transactions:', error);
      setTransactions([]);
    } finally {
      setLoading(false);
    }
  };



  const handleStatusChange = async (transactionId, currentStatus) => {
    // Handle overdue transactions with confirmation modal
    if (currentStatus === 'OVERDUE') {
      const transaction = transactions.find(t => t.transactionId === transactionId);
      setSelectedTransaction(transaction);
      setShowOverdueModal(true);
      return;
    }

    let newStatus = '';
    let apiMethod = null;

    if (currentStatus === 'PENDING') {
      newStatus = 'BORROWED';
      apiMethod = AdminApi.updateTransactionToBorrowed;
    } else if (currentStatus === 'BORROWED') {
      newStatus = 'RETURNED';
      apiMethod = AdminApi.updateTransactionToReturned;
    }

    if (newStatus && apiMethod) {
      try {
        const response = await apiMethod(transactionId);
        
        // Check if response has error code
        if (response.data && response.data.code !== 0) {
          const errorMessage = response.data.message || 'Có lỗi xảy ra khi cập nhật trạng thái';
          alert(`❌ Lỗi: ${errorMessage}`);
          return;
        }
        
        setTransactions(transactions.map(transaction => 
          transaction.transactionId === transactionId 
            ? { ...transaction, status: newStatus } 
            : transaction
        ));
        // Refresh data
        fetchTransactions();
        fetchAllTransactions();
      } catch (error) {
        console.error(`Error updating transaction status to ${newStatus}:`, error);
        
        // Handle API error response
        if (error.response && error.response.data) {
          const errorData = error.response.data;
          if (errorData.code && errorData.message) {
            alert(`❌ Lỗi (${errorData.code}): ${errorData.message}`);
          } else {
            alert('❌ Cập nhật trạng thái thất bại. Vui lòng thử lại.');
          }
        } else {
          alert('❌ Cập nhật trạng thái thất bại. Vui lòng thử lại.');
        }
      }
    }
  };

  const handleOverdueConfirmation = async () => {
    if (!selectedTransaction) return;

    try {
      const response = await AdminApi.updateTransactionToReturned(selectedTransaction.transactionId);
      
      // Check if response has error code
      if (response.data && response.data.code !== 0) {
        const errorMessage = response.data.message || 'Có lỗi xảy ra khi cập nhật trạng thái';
        alert(`❌ Lỗi: ${errorMessage}`);
        return;
      }
      
      setTransactions(transactions.map(transaction => 
        transaction.transactionId === selectedTransaction.transactionId 
          ? { ...transaction, status: 'RETURNED' } 
          : transaction
      ));
      // Refresh data
      fetchTransactions();
      fetchAllTransactions();
      // Close modal
      setShowOverdueModal(false);
      setSelectedTransaction(null);
    } catch (error) {
      console.error('Error updating overdue transaction:', error);
      
      // Handle API error response
      if (error.response && error.response.data) {
        const errorData = error.response.data;
        if (errorData.code && errorData.message) {
          alert(`❌ Lỗi (${errorData.code}): ${errorData.message}`);
        } else {
          alert('❌ Cập nhật trạng thái thất bại. Vui lòng thử lại.');
        }
      } else {
        alert('❌ Cập nhật trạng thái thất bại. Vui lòng thử lại.');
      }
    }
  };

  const handleOverdueCancel = () => {
    setShowOverdueModal(false);
    setSelectedTransaction(null);
  };

  const handleStatusFilterChange = (status) => {
    setStatusFilter(status);
    setCurrentPage(1);
    setSearchTerm('');
  };

  const handlePageChange = (page) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const getTransactionStats = () => {
    const pending = allTransactions.filter(t => t.status === 'PENDING').length;
    const borrowed = allTransactions.filter(t => t.status === 'BORROWED').length;
    const returned = allTransactions.filter(t => t.status === 'RETURNED').length;
    const cancelled = allTransactions.filter(t => t.status === 'CANCELLED').length;
    const overdue = allTransactions.filter(t => t.status === 'OVERDUE').length;
    const total = allTransactions.length;
    
    return { pending, borrowed, returned, cancelled, overdue, total };
  };

  const stats = getTransactionStats();

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleDateString('vi-VN');
  };

  const formatDateTime = (dateString) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleString('vi-VN');
  };

  const formatCurrency = (amount) => {
    if (!amount || amount === 0) return '0 VND';
    return new Intl.NumberFormat('vi-VN', {
      style: 'currency',
      currency: 'VND'
    }).format(amount);
  };

  // Calculate overdue fee if not provided by API
  const calculateOverdueFee = (transaction) => {
    if (transaction.overdueFee && transaction.overdueFee > 0) {
      return transaction.overdueFee;
    }
    
    if (transaction.status !== 'OVERDUE' || !transaction.dueDate) {
      return 0;
    }

    const currentDate = new Date();
    const dueDate = new Date(transaction.dueDate);
    
    if (currentDate <= dueDate) {
      return 0;
    }

    const overdueDays = Math.ceil((currentDate - dueDate) / (1000 * 60 * 60 * 24));
    const feePerDay = 5000; // 5,000 VND per day
    
    return overdueDays * feePerDay;
  };

  // Get display overdue fee for transaction
  const getOverdueFeeDisplay = (transaction) => {
    const fee = calculateOverdueFee(transaction);
    return formatCurrency(fee);
  };

  // Get overdue days for display
  const getOverdueDays = (transaction) => {
    if (transaction.status !== 'OVERDUE' || !transaction.dueDate) {
      return 0;
    }

    const currentDate = new Date();
    const dueDate = new Date(transaction.dueDate);
    
    if (currentDate <= dueDate) {
      return 0;
    }

    return Math.ceil((currentDate - dueDate) / (1000 * 60 * 60 * 24));
  };

  // Get column headers based on status filter
  const getTableHeaders = () => {
    switch (statusFilter) {
      case 'PENDING':
        return [
          { key: 'id', icon: '🆔', label: 'ID Giao dịch' },
          { key: 'book', icon: '📖', label: 'Thông tin sách' },
          { key: 'user', icon: '👤', label: 'Người mượn' },
          { key: 'borrowDate', icon: '📅', label: 'Ngày đặt trước' },
          { key: 'pickupDeadline', icon: '⏰', label: 'Hạn lấy sách' },
          { key: 'actions', icon: '⚙️', label: 'Thao tác' }
        ];
      case 'BORROWED':
        return [
          { key: 'id', icon: '🆔', label: 'ID Giao dịch' },
          { key: 'book', icon: '📖', label: 'Thông tin sách' },
          { key: 'user', icon: '👤', label: 'Người mượn' },
          { key: 'borrowDateOffline', icon: '📅', label: 'Ngày mượn thực tế' },
          { key: 'dueDate', icon: '⏰', label: 'Hạn trả' },
          { key: 'actions', icon: '⚙️', label: 'Thao tác' }
        ];
      case 'RETURNED':
        return [
          { key: 'id', icon: '🆔', label: 'ID Giao dịch' },
          { key: 'book', icon: '📖', label: 'Thông tin sách' },
          { key: 'user', icon: '👤', label: 'Người mượn' },
          { key: 'borrowDateOffline', icon: '📅', label: 'Ngày mượn' },
          { key: 'returnDate', icon: '✅', label: 'Ngày trả' },
          { key: 'actions', icon: '⚙️', label: 'Thao tác' }
        ];
      case 'OVERDUE':
        return [
          { key: 'id', icon: '🆔', label: 'ID Giao dịch' },
          { key: 'book', icon: '📖', label: 'Thông tin sách' },
          { key: 'user', icon: '👤', label: 'Người mượn' },
          { key: 'dueDate', icon: '⏰', label: 'Hạn trả' },
          { key: 'actions', icon: '⚙️', label: 'Thao tác' }
        ];
      case 'CANCELLED':
        return [
          { key: 'id', icon: '🆔', label: 'ID Giao dịch' },
          { key: 'book', icon: '📖', label: 'Thông tin sách' },
          { key: 'user', icon: '👤', label: 'Người mượn' },
          { key: 'borrowDate', icon: '📅', label: 'Ngày đặt trước' },
          { key: 'pickupDeadline', icon: '⏰', label: 'Hạn lấy sách' },
          { key: 'actions', icon: '⚙️', label: 'Thao tác' }
        ];
      default:
        return [
          { key: 'id', icon: '🆔', label: 'ID Giao dịch' },
          { key: 'book', icon: '📖', label: 'Thông tin sách' },
          { key: 'user', icon: '👤', label: 'Người mượn' },
          { key: 'borrowDate', icon: '📅', label: 'Ngày mượn' },
          { key: 'dueDate', icon: '⏰', label: 'Hạn trả' },
          { key: 'actions', icon: '⚙️', label: 'Thao tác' }
        ];
    }
  };

  // Render table cell content based on column key and status
  const renderTableCell = (transaction, columnKey) => {
    switch (columnKey) {
      case 'id':
        return (
          <td className={styles.idCell}>
            <span className={styles.transactionId}>#{transaction.transactionId}</span>
          </td>
        );
      case 'book':
        return (
          <td className={styles.bookInfoCell}>
            <div className={styles.bookInfo}>
              <h4 className={styles.bookTitle}>{transaction.book.title}</h4>
              <p className={styles.bookAuthor}>Tác giả: {transaction.book.author}</p>
            </div>
          </td>
        );
      case 'user':
        return (
          <td className={styles.userCell}>
            <div className={styles.userInfo}>
              <div className={styles.userAvatar}>
                {transaction.username?.charAt(0).toUpperCase() || 'U'}
              </div>
              <span className={styles.username}>
                {transaction.username || 'N/A'}
              </span>
            </div>
          </td>
        );
      case 'borrowDate':
        return (
          <td className={styles.dateCell}>
            <span className={styles.date}>{formatDateTime(transaction.borrowDate)}</span>
          </td>
        );
      case 'borrowDateOffline':
        return (
          <td className={styles.dateCell}>
            <span className={styles.date}>
              {transaction.borrowDateOffline ? formatDateTime(transaction.borrowDateOffline) : '-'}
            </span>
          </td>
        );
      case 'dueDate':
        return (
          <td className={styles.dateCell}>
            <span className={styles.date}>{formatDateTime(transaction.dueDate)}</span>
          </td>
        );
      case 'returnDate':
        return (
          <td className={styles.dateCell}>
            <span className={styles.date}>
              {transaction.returnDate ? formatDateTime(transaction.returnDate) : '-'}
            </span>
          </td>
        );
      case 'pickupDeadline':
        return (
          <td className={styles.dateCell}>
            <span className={styles.date}>
              {transaction.pickupDeadline ? formatDateTime(transaction.pickupDeadline) : '-'}
            </span>
          </td>
        );
      case 'overdueFee':
        return (
          <td className={styles.feeCell}>
            <span className={`${styles.fee} ${transaction.overdueFee > 0 ? styles.hasOverdueFee : ''}`}>
              {getOverdueFeeDisplay(transaction)}
            </span>
          </td>
        );
      case 'status':
        return (
          <td className={styles.statusCell}>
            <span className={`${styles.statusBadge} ${styles[statusConfig[transaction.status].color]}`}>
              <span className={styles.statusIcon}>
                {statusConfig[transaction.status].icon}
              </span>
              {statusConfig[transaction.status].label}
            </span>
          </td>
        );
      case 'actions':
        return (
          <td className={styles.actionsCell}>
            {statusConfig[transaction.status].action && (
              <button
                className={`${styles.actionButton} ${styles[statusConfig[transaction.status].color]}`}
                onClick={() => handleStatusChange(transaction.transactionId, transaction.status)}
              >
                {statusConfig[transaction.status].action}
              </button>
            )}
          </td>
        );
      default:
        return <td>-</td>;
    }
  };

  const renderPagination = () => {
    if (totalPages <= 1) return null;

    const pages = [];
    const maxVisiblePages = 5;
    
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    
    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    // Previous button
    if (currentPage > 1) {
      pages.push(
        <button
          key="prev"
          className={styles.paginationButton}
          onClick={() => handlePageChange(currentPage - 1)}
        >
          ‹ Trước
        </button>
      );
    }

    // Page numbers
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <button
          key={i}
          className={`${styles.paginationButton} ${
            currentPage === i ? styles.active : ''
          }`}
          onClick={() => handlePageChange(i)}
        >
          {i}
        </button>
      );
    }

    // Next button
    if (currentPage < totalPages) {
      pages.push(
        <button
          key="next"
          className={styles.paginationButton}
          onClick={() => handlePageChange(currentPage + 1)}
        >
          Tiếp ›
        </button>
      );
    }

    return pages;
  };

  if (loading && transactions.length === 0) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.loadingSpinner}></div>
        <p className={styles.loadingMessage}>Đang tải danh sách giao dịch...</p>
      </div>
    );
  }

  return (
    <div className={styles.transactionManagementContainer}>
      {/* Header Section */}
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h2 className={styles.title}>
            <span className={styles.titleIcon}>📋</span>
            Quản lý giao dịch
          </h2>
          <p className={styles.subtitle}>
            Theo dõi và quản lý trạng thái mượn trả sách trong hệ thống
          </p>
        </div>
        <button className={styles.refreshButton} onClick={() => {
          fetchTransactions();
          fetchAllTransactions();
        }}>
          <span className={styles.refreshIcon}>🔄</span>
          Làm mới
        </button>
      </div>



      {/* Status Filter Tabs */}
      <div className={styles.statusTabs}>
        {Object.entries(statusConfig).map(([status, config]) => (
          <button
            key={status}
            className={`${styles.statusTab} ${
              statusFilter === status ? styles.active : ''
            } ${styles[config.color]}`}
            onClick={() => handleStatusFilterChange(status)}
          >
            <span className={styles.tabIcon}>{config.icon}</span>
            <div className={styles.tabContent}>
              <span className={styles.tabLabel}>{config.label}</span>
              <span className={styles.tabCount}>
                {status === 'PENDING' && stats.pending}
                {status === 'BORROWED' && stats.borrowed}
                {status === 'RETURNED' && stats.returned}
                {status === 'CANCELLED' && stats.cancelled}
                {status === 'OVERDUE' && stats.overdue}
              </span>
            </div>
          </button>
        ))}
      </div>

      {/* Search Section */}
      <div className={styles.searchContainer}>
        <span className={styles.searchIcon}>🔍</span>
        <input
          type="text"
          placeholder="Tìm kiếm theo ID giao dịch, tên sách hoặc người dùng..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className={styles.searchInput}
        />
      </div>

      {/* Results Info */}
      <div className={styles.resultsInfo}>
        <span className={styles.resultsCount}>
          {searchTerm ? (
            `Hiển thị ${transactions.length} kết quả tìm kiếm từ ${totalElements} giao dịch`
          ) : (
            `Hiển thị ${transactions.length} / ${totalElements} giao dịch`
          )}
        </span>
        {!searchTerm && totalPages > 1 && (
          <span className={styles.pageInfo}>
            Trang {currentPage} / {totalPages}
          </span>
        )}
      </div>

      {/* Transactions Table */}
      <div className={styles.tableContainer}>
        {transactions.length > 0 ? (
          <table className={styles.transactionTable}>
            <thead>
              <tr>
                {getTableHeaders().map((header, index) => (
                  <th key={index}>
                    <span className={styles.headerIcon}>{header.icon}</span>
                    {header.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {transactions.map(transaction => (
                <tr key={transaction.transactionId} className={styles.transactionRow}>
                  {getTableHeaders().map((header, index) => renderTableCell(transaction, header.key))}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>📋</div>
            <h3 className={styles.emptyTitle}>Không tìm thấy giao dịch nào</h3>
            <p className={styles.emptyDescription}>
              {searchTerm 
                ? `Không có giao dịch nào phù hợp với từ khóa "${searchTerm}".`
                : `Không có giao dịch nào với trạng thái "${statusConfig[statusFilter].label}".`
              }
            </p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {!searchTerm && totalPages > 1 && (
        <div className={styles.paginationContainer}>
          <div className={styles.paginationInfo}>
            Trang {currentPage} / {totalPages} - Tổng {totalElements} giao dịch
          </div>
          <div className={styles.pagination}>
            {renderPagination()}
          </div>
        </div>
      )}

      {/* Overdue Fee Confirmation Modal */}
      {showOverdueModal && selectedTransaction && (
        <div className={styles.modalOverlay}>
          <div className={styles.modal}>
            <div className={styles.modalHeader}>
              <h3 className={styles.modalTitle}>
                <span className={styles.modalIcon}>⚠️</span>
                Xác nhận trả sách quá hạn
              </h3>
              <button 
                className={styles.modalCloseButton}
                onClick={handleOverdueCancel}
              >
                ✕
              </button>
            </div>
            
            <div className={styles.modalBody}>
              <div className={styles.transactionInfo}>
                <h4 className={styles.bookTitleModal}>
                  📖 {selectedTransaction.book.title}
                </h4>
                <p className={styles.bookAuthorModal}>
                  Tác giả: {selectedTransaction.book.author}
                </p>
                <p className={styles.borrowerModal}>
                  👤 Người mượn: {selectedTransaction.username}
                </p>
                <p className={styles.dueDateModal}>
                  ⏰ Hạn trả: {formatDateTime(selectedTransaction.dueDate)}
                </p>
              </div>

              <div className={styles.feeInfo}>
                <div className={styles.feeLabel}>Phí phạt quá hạn:</div>
                <div className={styles.feeAmount}>
                  {getOverdueFeeDisplay(selectedTransaction)}
                </div>
                <div className={styles.overdueDays}>
                  Quá hạn: {getOverdueDays(selectedTransaction)} ngày
                </div>
              </div>

              <div className={styles.confirmationText}>
                <p>⚠️ Người mượn đã trả tiền phí phạt và trả sách?</p>
                <p className={styles.warningText}>
                  Hành động này sẽ cập nhật trạng thái giao dịch thành "Đã trả sách" 
                  và không thể hoàn tác.
                </p>
              </div>
            </div>

            <div className={styles.modalFooter}>
              <button 
                className={styles.cancelButton}
                onClick={handleOverdueCancel}
              >
                Hủy
              </button>
              <button 
                className={styles.confirmButton}
                onClick={handleOverdueConfirmation}
              >
                Xác nhận đã trả tiền & sách
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default TransactionManagement;
