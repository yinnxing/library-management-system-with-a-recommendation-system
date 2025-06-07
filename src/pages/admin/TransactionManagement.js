import React, { useState, useEffect } from 'react';
import AdminApi from '../../api/AdminApi';
import styles from './TransactionManagement.module.css'; 
import Chart from '../../components/admin/chart/Chart';

const TransactionManagement = () => {
  const [transactions, setTransactions] = useState([]);
  const [allTransactions, setAllTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('PENDING');
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [totalElements, setTotalElements] = useState(0);
  const [chartData, setChartData] = useState([]);

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

  useEffect(() => {
    fetchChartData();
  }, [statusFilter, allTransactions]);

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
            transaction.user?.username?.toLowerCase().includes(searchTerm.toLowerCase())
          );
        }
        
        setTransactions(filteredTransactions);
        setTotalPages(result.totalPages);
        setTotalElements(result.totalElements);
      }
    } catch (error) {
      console.error('Error fetching transactions:', error);
      setTransactions([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchChartData = async () => {
    try {
      const transactionsForChart = allTransactions.filter(t => t.status === statusFilter);
      const formattedData = transactionsForChart.reduce((acc, transaction) => {
        let date;

        if (statusFilter === 'PENDING') {
          date = transaction.borrowDate;
        } else if (statusFilter === 'BORROWED') {
          date = transaction.dueDate;
        } else if (statusFilter === 'RETURNED') {
          date = transaction.returnDate;
        } else {
          date = transaction.borrowDate;
        }

        if (date) {
          date = date.split('T')[0];
          if (!acc[date]) {
            acc[date] = 0;
          }
          acc[date]++;
        }

        return acc;
      }, {});

      const sortedChartData = Object.keys(formattedData)
        .map(date => ({
          date,
          count: formattedData[date]
        }))
        .sort((a, b) => new Date(a.date) - new Date(b.date));

      setChartData(sortedChartData);
    } catch (error) {
      console.error('Error fetching chart data:', error);
    }
  };

  const handleStatusChange = async (transactionId, currentStatus) => {
    let newStatus = '';
    let apiMethod = null;

    if (currentStatus === 'PENDING') {
      newStatus = 'BORROWED';
      apiMethod = AdminApi.updateTransactionToBorrowed;
    } else if (currentStatus === 'BORROWED' || currentStatus === 'OVERDUE') {
      newStatus = 'RETURNED';
      apiMethod = AdminApi.updateTransactionToReturned;
    }

    if (newStatus && apiMethod) {
      try {
        await apiMethod(transactionId);
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
        alert('Cập nhật trạng thái thất bại. Vui lòng thử lại.');
      }
    }
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

      {/* Stats Cards */}
      <div className={styles.statsContainer}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>📊</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.total}</h3>
            <p className={styles.statLabel}>Tổng giao dịch</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>⏳</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.pending}</h3>
            <p className={styles.statLabel}>Chờ mượn</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>📖</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.borrowed}</h3>
            <p className={styles.statLabel}>Đang mượn</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>✅</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.returned}</h3>
            <p className={styles.statLabel}>Đã trả</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>⚠️</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.overdue}</h3>
            <p className={styles.statLabel}>Quá hạn</p>
          </div>
        </div>
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
                <th>
                  <span className={styles.headerIcon}>🆔</span>
                  ID Giao dịch
                </th>
                <th>
                  <span className={styles.headerIcon}>📖</span>
                  Thông tin sách
                </th>
                <th>
                  <span className={styles.headerIcon}>👤</span>
                  Người mượn
                </th>
                <th>
                  <span className={styles.headerIcon}>📅</span>
                  Ngày mượn
                </th>
                <th>
                  <span className={styles.headerIcon}>⏰</span>
                  Hạn trả
                </th>
                <th>
                  <span className={styles.headerIcon}>📋</span>
                  Trạng thái
                </th>
                <th>
                  <span className={styles.headerIcon}>⚙️</span>
                  Thao tác
                </th>
              </tr>
            </thead>
            <tbody>
              {transactions.map(transaction => (
                <tr key={transaction.transactionId} className={styles.transactionRow}>
                  <td className={styles.idCell}>
                    <span className={styles.transactionId}>#{transaction.transactionId}</span>
                  </td>
                  <td className={styles.bookInfoCell}>
                    <div className={styles.bookInfo}>
                      <h4 className={styles.bookTitle}>{transaction.book.title}</h4>
                      <p className={styles.bookAuthor}>Tác giả: {transaction.book.author}</p>
                    </div>
                  </td>
                  <td className={styles.userCell}>
                    <div className={styles.userInfo}>
                      <div className={styles.userAvatar}>
                        {transaction.user?.username?.charAt(0).toUpperCase() || 'U'}
                      </div>
                      <span className={styles.username}>
                        {transaction.user?.username || 'N/A'}
                      </span>
                    </div>
                  </td>
                  <td className={styles.dateCell}>
                    <span className={styles.date}>{formatDate(transaction.borrowDate)}</span>
                  </td>
                  <td className={styles.dateCell}>
                    <span className={styles.date}>{formatDate(transaction.dueDate)}</span>
                  </td>
                  <td className={styles.statusCell}>
                    <span className={`${styles.statusBadge} ${styles[statusConfig[transaction.status].color]}`}>
                      <span className={styles.statusIcon}>
                        {statusConfig[transaction.status].icon}
                      </span>
                      {statusConfig[transaction.status].label}
                    </span>
                  </td>
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

      {/* Chart Section */}
      <div className={styles.chartSection}>
        <div className={styles.chartHeader}>
          <h3 className={styles.chartTitle}>
            <span className={styles.chartIcon}>📈</span>
            Biểu đồ giao dịch - {statusConfig[statusFilter].label}
          </h3>
          <p className={styles.chartSubtitle}>
            Thống kê số lượng giao dịch theo thời gian
          </p>
        </div>
        <Chart 
          chartTitle={`Giao dịch ${statusConfig[statusFilter].label}`}
          label="Số lượng giao dịch" 
          chartColor="rgba(75,192,192,1)" 
          data={chartData} 
        />
      </div>
    </div>
  );
};

export default TransactionManagement;
