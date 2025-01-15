import React, { useState, useEffect } from 'react';
import AdminApi from '../../api/AdminApi';
import styles from './TransactionManagement.module.css'; 
import Chart from '../../components/admin/chart/Chart';

const TransactionManagement = () => {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('PENDING');
  const [pagination, setPagination] = useState({ page: 1, size: 10 });
  const [chartData, setChartData] = useState([]);

useEffect(() => {
  const fetchChartData = async () => {
    try {
      const response = await AdminApi.getTransactions( { page: 1, size: 100 });
      const transactionsForChart = response.data.result.content;
      const formattedData = transactionsForChart.reduce((acc, transaction) => {
        let date;

        if (statusFilter === 'PENDING') {
          date = transaction.borrowDate;
        } else if (statusFilter === 'BORROWED') {
          date = transaction.dueDate;
        } else if (statusFilter === 'RETURNED') {
          date = transaction.returnDate;
        }

        if (date) {
          date = date.split('T')[0]; // Lấy phần ngày (YYYY-MM-DD)
          if (!acc[date]) {
            acc[date] = 0;
          }
          acc[date]++;
        }

        return acc;
      }, {});

      // Chuyển đổi dữ liệu sang mảng và sắp xếp
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

  fetchChartData();
}, [statusFilter]); // Gọi lại mỗi khi tab (statusFilter) thay đổi


  useEffect(() => {
    setLoading(true);
    AdminApi.getTransactions({ status: statusFilter }, pagination)
      .then(response => {
        setTransactions(response.data.result.content);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching transactions:', error);
        setLoading(false);
      });
  }, [statusFilter, pagination]);

  const handleStatusChange = (transactionId, currentStatus) => {
    let newStatus = '';
    if (currentStatus === 'PENDING') {
      newStatus = 'BORROWED';
    } else if (currentStatus === 'BORROWED') {
      newStatus = 'RETURNED';
    }

    if (newStatus) {
      // Chọn API phù hợp để cập nhật trạng thái
      const apiMethod = newStatus === 'BORROWED' 
        ? AdminApi.updateTransactionToBorrowed
        : AdminApi.updateTransactionToReturned;

      apiMethod(transactionId)
        .then(() => {
          setTransactions(transactions.map(transaction => 
            transaction.transactionId === transactionId ? { ...transaction, status: newStatus } : transaction
          ));
        })
        .catch(error => {
          console.error(`Error updating transaction status to ${newStatus}:`, error);
        });
    }
  };

  return (
    <div className={styles.transactionManagement}>
      <h2 className={styles.pageTitle}>Transaction Management</h2>

      <div className={styles.statusFilter}>
        {['PENDING', 'BORROWED', 'RETURNED'].map(status => (
          <button
            key={status}
            className={`${styles.filterButton} ${status === statusFilter ? styles.activeFilter : ''}`}
            onClick={() => setStatusFilter(status)}
          >
            {status}
          </button>
        ))}
      </div>

      <table className={styles.table}>
        <thead>
          <tr>
            <th>Transaction ID</th>
            <th>Book Title</th>
            <th>Borrow Date</th>
            <th>Due Date</th>
            <th>Return Date</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr>
              <td colSpan="6" className={styles.loadingSpinner}>Loading...</td>
            </tr>
          ) : (
            transactions.map(transaction => (
              <tr key={transaction.transactionId}>
                <td>{transaction.transactionId}</td>
                <td>{transaction.book.title}</td>
                <td>{transaction.borrowDate}</td>
                <td>{transaction.dueDate}</td>
                <td>{transaction.returnDate}</td>
                <td>{transaction.status}</td>
                <td>
                  {transaction.status === 'PENDING' && (
                    <button
                      className={styles.updateButton}
                      onClick={() => handleStatusChange(transaction.transactionId, 'PENDING')}
                    >
                      Mark as Borrowed
                    </button>
                  )}
                  {transaction.status === 'BORROWED' && (
                    <button
                      className={styles.updateButton}
                      onClick={() => handleStatusChange(transaction.transactionId, 'BORROWED')}
                    >
                      Mark as Returned
                    </button>
                  )}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>

      <div className={styles.pagination}>
        <button
          disabled={pagination.page <= 1}
          onClick={() => setPagination(prev => ({ ...prev, page: prev.page - 1 }))}
        >
          Previous
        </button>
        <button
          disabled={transactions.length < pagination.size}
          onClick={() => setPagination(prev => ({ ...prev, page: prev.page + 1 }))}
        >
          Next
        </button>
      </div>
      {/* Pass formatted chart data to Chart */}
      <Chart 
          chartTitle="Transactions Over Time" 
          label="Transactions" 
          chartColor="rgba(75,192,192,1)" 
          data={chartData} 
      />
    </div>
  );
};

export default TransactionManagement;
