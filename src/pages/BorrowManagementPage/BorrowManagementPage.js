import React, { useState, useEffect} from 'react';
import '../../styles/design-system.css';
import styles from './BorrowManagementPage.module.css';
import BookIcon from '@mui/icons-material/Book';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import WarningIcon from '@mui/icons-material/Warning';
import RefreshIcon from '@mui/icons-material/Refresh';
import InfoIcon from '@mui/icons-material/Info';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import MonetizationOnIcon from '@mui/icons-material/MonetizationOn';
import RuleIcon from '@mui/icons-material/Rule';
import UserApi from '../../api/UserApi';
import { useUser } from '../../contexts/UserContext'; 
import Book from '../../components/TransactionBook/TransactionBook'; 

const BorrowManagementPage = () => {
  const [selectedTab, setSelectedTab] = useState('PENDING');
  const { user } = useUser();
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({
    PENDING: 0,
    BORROWED: 0,
    RETURNED: 0,
    CANCELLED: 0,
    OVERDUE: 0
  });

  // Hàm gọi API tương ứng theo trạng thái
  const fetchBooksByTab = async () => {
    setLoading(true);
    try {
      // Use unified API call with status filter
      const response = await UserApi.getTransactionsByStatus(user.userId, selectedTab);
      
      const fetchedBooks = response.data.map((transaction) => ({
        transactionId: transaction.transactionId,
        book: transaction.book,
        borrowDate: transaction.borrowDate,
        dueDate: transaction.dueDate,
        returnDate: transaction.returnDate,
        pickupDeadline: transaction.pickupDeadline,
        overdueFee: transaction.overdueFee,
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

  // Fetch statistics for all tabs
  const fetchStats = async () => {
    try {
      const [pendingRes, borrowedRes, returnedRes, cancelledRes, overdueRes] = await Promise.all([
        UserApi.getTransactionsByStatus(user.userId, 'PENDING'),
        UserApi.getTransactionsByStatus(user.userId, 'BORROWED'),
        UserApi.getTransactionsByStatus(user.userId, 'RETURNED'),
        UserApi.getTransactionsByStatus(user.userId, 'CANCELLED'),
        UserApi.getTransactionsByStatus(user.userId, 'OVERDUE')
      ]);
      
      setStats({
        PENDING: pendingRes.data.length,
        BORROWED: borrowedRes.data.length,
        RETURNED: returnedRes.data.length,
        CANCELLED: cancelledRes.data.length,
        OVERDUE: overdueRes.data.length
      });
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const handleRefresh = () => {
    fetchBooksByTab();
    fetchStats();
  };

  // Gọi API khi thay đổi tab
  useEffect(() => {
    if (user?.userId) {
    fetchBooksByTab();
      fetchStats();
    }
  }, [selectedTab, user?.userId]);

  const tabConfig = [
    { 
      label: 'Chờ nhận sách', 
      value: 'PENDING', 
      Icon: HourglassEmptyIcon,
      description: 'Yêu cầu mượn sách đang chờ nhận tại thư viện',
      color: 'warning'
    },
    { 
      label: 'Đang mượn', 
      value: 'BORROWED', 
      Icon: BookIcon,
      description: 'Sách bạn đang mượn',
      color: 'primary'
    },
    { 
      label: 'Đã trả', 
      value: 'RETURNED', 
      Icon: CheckCircleIcon,
      description: 'Lịch sử sách đã trả',
      color: 'success'
    },
    { 
      label: 'Đã hủy', 
      value: 'CANCELLED', 
      Icon: CancelIcon,
      description: 'Giao dịch bị hủy do quá hạn nhận sách',
      color: 'error'
    },
    { 
      label: 'Quá hạn', 
      value: 'OVERDUE', 
      Icon: WarningIcon,
      description: 'Sách đã quá hạn trả',
      color: 'error'
    },
  ];

  const currentTab = tabConfig.find(tab => tab.value === selectedTab);

  const borrowingPolicies = [
    {
      icon: <BookIcon />,
      title: "Số lượng mượn tối đa",
      description: "Mỗi độc giả được mượn tối đa 5 cuốn sách cùng một lúc",
      details: "Bao gồm cả sách giáo khoa và sách tham khảo"
    },
    {
      icon: <AccessTimeIcon />,
      title: "Thời gian mượn",
      description: "Thời gian mượn sách là 14 ngày kể từ ngày mượn",
      details: "Có thể gia hạn thêm 7 ngày nếu không có người đặt trước"
    },
    {
      icon: <MonetizationOnIcon />,
      title: "Phí phạt trễ hạn",
      description: "Phí phạt 5.000 VNĐ/ngày cho mỗi cuốn sách trả trễ",
      details: "Tính từ ngày hết hạn đến ngày trả sách thực tế"
    },
    {
      icon: <RuleIcon />,
      title: "Quy định bảo quản",
      description: "Giữ gìn sách sạch sẽ, không làm hỏng hoặc mất sách",
      details: "Nếu làm hỏng sẽ phải bồi thường theo giá trị sách"
    }
  ];

  return (
    <div className={styles.container}>
      {/* Header Section */}
      <div className={styles.headerSection}>
        <div className={styles.titleContainer}>
          <h1 className={styles.pageTitle}>
            <span className={styles.titleIcon}>📚</span>
            Quản Lý Mượn Sách
          </h1>
          <p className={styles.pageSubtitle}>
            Theo dõi và quản lý tất cả các giao dịch mượn sách của bạn
          </p>
        </div>
      </div>

      {/* Borrowing Policy Section */}
      <div className={styles.policySection}>
        <div className={styles.policyHeader}>
          <h2 className={styles.policyTitle}>
            <InfoIcon className={styles.policyIcon} />
            Chính Sách Mượn Sách
          </h2>
          <p className={styles.policySubtitle}>
            Vui lòng đọc kỹ các quy định dưới đây để đảm bảo việc mượn sách diễn ra thuận lợi
          </p>
        </div>
        
        <div className={styles.policyGrid}>
          {borrowingPolicies.map((policy, index) => (
            <div key={index} className={styles.policyCard}>
              <div className={styles.policyCardIcon}>
                {policy.icon}
              </div>
              <div className={styles.policyCardContent}>
                <h3 className={styles.policyCardTitle}>{policy.title}</h3>
                <p className={styles.policyCardDescription}>{policy.description}</p>
                <p className={styles.policyCardDetails}>{policy.details}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Stats Overview */}
      {/* <div className={styles.statsSection}>
        {tabConfig.map(({ label, value, Icon, color }) => (
          <div key={value} className={`${styles.statCard} ${styles[color]}`}>
            <div className={styles.statIcon}>
              <Icon />
            </div>
            <div className={styles.statContent}>
              <div className={styles.statNumber}>{stats[value]}</div>
              <div className={styles.statLabel}>{label}</div>
            </div>
          </div>
        ))}
      </div> */}

      {/* Main Content */}
      <div className={styles.mainContent}>
        {/* Tab Navigation */}
        <div className={styles.tabNavigation}>
          {tabConfig.map(({ label, value, Icon, description, color }) => (
          <button
            key={value}
              className={`${styles.tabButton} ${selectedTab === value ? styles.activeTab : ''} ${styles[color]}`}
            onClick={() => setSelectedTab(value)}
          >
              <div className={styles.tabIcon}>
                <Icon />
              </div>
              <div className={styles.tabContent}>
                <div className={styles.tabLabel}>{label}</div>
                <div className={styles.tabDescription}>{description}</div>
                <div className={styles.tabCount}>{stats[value]} cuốn</div>
              </div>
          </button>
        ))}
      </div>

        {/* Content Area */}
        <div className={styles.contentArea}>
          <div className={styles.contentHeader}>
            <h2 className={styles.contentTitle}>
              <currentTab.Icon className={styles.contentIcon} />
              {currentTab.label}
            </h2>
            <div className={styles.contentMeta}>
              {stats[selectedTab]} cuốn sách
            </div>
          </div>

          <div className={styles.contentBody}>
        {loading ? (
              <div className={styles.loadingContainer}>
                <div className={styles.loadingSpinner}></div>
                <p className={styles.loadingText}>Đang tải dữ liệu...</p>
              </div>
        ) : books.length > 0 ? (
              <div className={styles.bookList}>
            {books.map((transaction) => (
                <Book key={transaction.transactionId} transaction={transaction} /> 
            ))}
              </div>
        ) : (
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>
                  <currentTab.Icon />
                </div>
                <h3 className={styles.emptyTitle}>Không có sách nào</h3>
                <p className={styles.emptyDescription}>
                  {selectedTab === 'PENDING' && 'Bạn chưa có yêu cầu mượn sách nào đang chờ nhận tại thư viện.'}
                  {selectedTab === 'BORROWED' && 'Bạn hiện tại không đang mượn sách nào.'}
                  {selectedTab === 'RETURNED' && 'Bạn chưa trả sách nào.'}
                  {selectedTab === 'CANCELLED' && 'Bạn chưa có giao dịch nào bị hủy.'}
                  {selectedTab === 'OVERDUE' && 'Bạn không có sách nào đang quá hạn.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BorrowManagementPage;
