import React, { useEffect, useState } from 'react';
import AdminApi from '../../api/AdminApi'; 
import styles from './UserManagement.module.css'; 

const UserManagement = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const response = await AdminApi.getUsers();
      if (response.data.code === 0) {
        // Filter to only show USER role users
        const userRoleUsers = response.data.result.filter(user => user.role === 'USER');
        setUsers(userRoleUsers);
      }
    } catch (error) {
      console.error('Error fetching users:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSearch;
  });

  const getUserStats = () => {
    const totalUsers = users.length;
    return { totalUsers };
  };

  const stats = getUserStats();

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.loadingSpinner}></div>
        <p className={styles.loadingMessage}>Đang tải danh sách người dùng...</p>
      </div>
    );
  }

  return (
    <div className={styles.userManagementContainer}>
      {/* Header Section */}
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h2 className={styles.title}>
            <span className={styles.titleIcon}>👤</span>
            Quản lý người dùng
          </h2>
          <p className={styles.subtitle}>
            Xem thông tin người dùng trong hệ thống
          </p>
        </div>
        <button className={styles.refreshButton} onClick={fetchUsers}>
          <span className={styles.refreshIcon}>🔄</span>
          Làm mới
        </button>
      </div>

      {/* Stats Card */}
      <div className={styles.statsContainer}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>👤</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.totalUsers}</h3>
            <p className={styles.statLabel}>Tổng người dùng</p>
          </div>
        </div>
      </div>

      {/* Search Section */}
      <div className={styles.filtersContainer}>
        <div className={styles.searchContainer}>
          <span className={styles.searchIcon}>🔍</span>
          <input
            type="text"
            placeholder="Tìm kiếm theo tên hoặc email..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={styles.searchInput}
          />
        </div>
      </div>

      {/* Results Info */}
      <div className={styles.resultsInfo}>
        <span className={styles.resultsCount}>
          Hiển thị {filteredUsers.length} / {users.length} người dùng
        </span>
      </div>

      {/* Users Table */}
      <div className={styles.tableContainer}>
        {filteredUsers.length > 0 ? (
          <table className={styles.userTable}>
            <thead>
              <tr>
                <th>
                  <span className={styles.headerIcon}>🆔</span>
                  ID
                </th>
                <th>
                  <span className={styles.headerIcon}>👤</span>
                  Tên người dùng
                </th>
                <th>
                  <span className={styles.headerIcon}>📧</span>
                  Email
                </th>
                <th>
                  <span className={styles.headerIcon}>📅</span>
                  Trạng thái
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredUsers.map((user) => (
                <tr key={user.userId} className={styles.userRow}>
                  <td className={styles.userIdCell}>
                    <span className={styles.userId}>#{user.userId}</span>
                  </td>
                  <td className={styles.usernameCell}>
                    <div className={styles.userInfo}>
                      <div className={styles.userAvatar}>
                        {user.username.charAt(0).toUpperCase()}
                      </div>
                      <span className={styles.username}>{user.username}</span>
                    </div>
                  </td>
                  <td className={styles.emailCell}>
                    <span className={styles.email}>{user.email}</span>
                  </td>
                  <td className={styles.statusCell}>
                    <span className={styles.statusBadge}>
                      <span className={styles.statusIcon}>✅</span>
                      Hoạt động
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>👤</div>
            <h3 className={styles.emptyTitle}>Không tìm thấy người dùng</h3>
            <p className={styles.emptyDescription}>
              Không có người dùng nào phù hợp với tiêu chí tìm kiếm của bạn.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default UserManagement;
