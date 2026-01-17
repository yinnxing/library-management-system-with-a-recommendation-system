import React from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom'; 
import styles from './AdminDashboard.module.css'; 
import { useNavigate } from 'react-router-dom';  

const AdminDashboard = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  const handleLogout = () => {
    localStorage.removeItem('adminToken');
    navigate('/login');
  };

  const navigationItems = [
    {
      path: '/admin/books',
      label: 'Quản lý sách',
      icon: '📚',
      description: 'Thêm, sửa, xóa sách'
    },
    {
      path: '/admin/users',
      label: 'Quản lý người dùng',
      icon: '👥',
      description: 'Quản lý tài khoản người dùng'
    },
    {
      path: '/admin/transactions',
      label: 'Quản lý giao dịch',
      icon: '📋',
      description: 'Theo dõi mượn trả sách'
    }
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <div className={styles.dashboardContainer}>
      {/* Sidebar */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>⚡</span>
            <h3 className={styles.sidebarTitle}>Harmony Admin</h3>
          </div>
          <p className={styles.sidebarSubtitle}>Bảng điều khiển quản trị</p>
        </div>

        <nav className={styles.nav}>
          <ul className={styles.navList}>
            {navigationItems.map((item) => (
              <li key={item.path} className={styles.navItem}>
                <Link 
                  to={item.path} 
                  className={`${styles.navLink} ${isActive(item.path) ? styles.active : ''}`}
                >
                  <span className={styles.navIcon}>{item.icon}</span>
                  <div className={styles.navContent}>
                    <span className={styles.navLabel}>{item.label}</span>
                    <span className={styles.navDescription}>{item.description}</span>
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <div className={styles.sidebarFooter}>
          <div className={styles.adminInfo}>
            <div className={styles.adminAvatar}>👤</div>
            <div className={styles.adminDetails}>
              <span className={styles.adminName}>Admin</span>
              <span className={styles.adminRole}>Quản trị viên</span>
            </div>
          </div>
          <button onClick={handleLogout} className={styles.logoutButton}>
            <span className={styles.logoutIcon}>🚪</span>
            Đăng xuất
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className={styles.mainContent}>
        <div className={styles.contentHeader}>
          <div className={styles.breadcrumb}>
            <span className={styles.breadcrumbHome}>🏠</span>
            <span className={styles.breadcrumbSeparator}>›</span>
            <span className={styles.breadcrumbCurrent}>
              {navigationItems.find(item => isActive(item.path))?.label || 'Bảng điều khiển'}
            </span>
          </div>

        </div>

        <div className={styles.contentBody}>
          <Outlet />
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
