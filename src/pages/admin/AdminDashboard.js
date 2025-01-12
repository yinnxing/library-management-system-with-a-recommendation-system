import React from 'react';
import { Link, Outlet } from 'react-router-dom'; 
import styles from './AdminDashboard.module.css'; 
import { useNavigate } from 'react-router-dom';  


const AdminDashboard = () => {
  const navigate = useNavigate();  
    const handleLogout = () => {
    localStorage.removeItem('adminToken');
    navigate('/login');
  };
 

  return (
    <div className={styles.dashboardContainer}>
      {/* Sidebar */}
      <div className={styles.sidebar}>
        <h3 className={styles.sidebarTitle}>Admin Dashboard</h3>
        <nav className={styles.nav}>
          <ul>
            <li><Link to="/admin/books" className={styles.navLink}>Book Management</Link></li>
            <li><Link to="/admin/users" className={styles.navLink}>User Management</Link></li>
            <li><Link to="/admin/transactions" className={styles.navLink}>Transaction Management</Link></li>
          </ul>
        </nav>
        {/* Nút đăng xuất */}
        <button onClick={handleLogout} className={styles.logoutButton}>Logout</button>
      </div>

      <div className={styles.mainContent}>
        <h2 className={styles.dashboardTitle}> </h2>

        {/* Outlet để render các trang con như Book Management, User Management... */}
        <Outlet />
      </div>
    </div>
  );
};

export default AdminDashboard;
