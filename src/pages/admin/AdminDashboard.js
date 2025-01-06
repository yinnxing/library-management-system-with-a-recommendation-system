import React from 'react';
import { Link, Outlet } from 'react-router-dom'; 
import styles from './AdminDashboard.module.css'; 
import Chart from '../../components/admin/chart/Chart';


const AdminDashboard = () => {
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
            <li><Link to="/admin/reports" className={styles.navLink}>Reports</Link></li>
          </ul>
        </nav>
      </div>

      <div className={styles.mainContent}>
        <h2 className={styles.dashboardTitle}> </h2>

        {/* Outlet để render các trang con như Book Management, User Management... */}
        <Outlet />
        
        <Chart 
          fetchDataUrl="/api/admin/transactions" 
          chartTitle="Transactions Over Time" 
          label="Transactions" 
          chartColor="rgba(75,192,192,1)" 
        />
      </div>
    </div>
  );
};

export default AdminDashboard;
