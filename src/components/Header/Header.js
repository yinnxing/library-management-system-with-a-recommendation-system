import React, { useState } from 'react';
import { Link, useNavigate, NavLink } from 'react-router-dom';
import Search from '../Search/Search';
import styles from './Header.module.css';
import { useUser } from '../../contexts/UserContext'; 
import Cookies from 'js-cookie';
import HeadlessTippy from '@tippyjs/react/headless';
import { Modal } from 'antd';
import UserApi from '../../api/UserApi';

const Header = () => {
  const { user, logout, isAuthenticated } = useUser();
  const navigate = useNavigate();
  const [isLoginModalVisible, setIsLoginModalVisible] = useState(false);
  const [visible, setVisible] = useState(false);

  const handleLogout = async () => {
    try {
      // Call logout API if available
      try {
        await UserApi.logout();
      } catch (error) {
        console.log('Logout API error, continuing with local logout:', error);
      }
      
      // Clean up local state
      logout();
      
      // Update UI state
      setVisible(false);
      
      // Navigate to home page
      navigate('/');
    } catch (error) {
      console.error('Logout error:', error);
      logout();
      navigate('/');
    }
  };

  const showLoginModal = () => {
    setIsLoginModalVisible(true);
  };

  const handleLoginModalCancel = () => {
    setIsLoginModalVisible(false);
  };

  const goToLogin = () => {
    setIsLoginModalVisible(false);
    navigate('/login');
  };

  const goToSignup = () => {
    setIsLoginModalVisible(false);
    navigate('/signup');
  };

  const goToProfile = () => {
    setVisible(false);
    navigate('/user-profile');
  };

  return (
    <header className={styles.header}>
      {/* Logo */}
      <div className={styles.logo}>
        <Link to="/">Harmony Library</Link>
      </div>

      {/* Search Bar */}
      <div className={styles.searchWrapper}>
        <Search />
      </div>

      {/* Navigation Menu */}
      <nav className={styles.nav}>
        <NavLink
          to="/favorite"
          className={({ isActive }) => isActive ? styles.activeLink : styles.button}
        >
          Danh sách yêu thích
        </NavLink>
        <NavLink
          to="/borrowBook"
          className={({ isActive }) => isActive ? styles.activeLink : styles.button}
        >
          Mượn sách
        </NavLink>
        <NavLink
          to="/advanced-search"
          className={({ isActive }) => isActive ? styles.activeLink : styles.button}
        >
          Advanced Search
        </NavLink>

        {/* Authentication UI - Shows login button or user menu based on auth state */}
        {!isAuthenticated ? (
          <button onClick={showLoginModal} className={styles.loginButton}>
            Đăng nhập / Đăng ký
          </button>
        ) : (
          <HeadlessTippy
            visible={visible} 
            onClickOutside={() => setVisible(false)} 
            placement="bottom"
            interactive={true} 
            arrow={false} 
            render={(attrs) => (
              <div
                className={styles.userProfileMenu}
                tabIndex="-1"
                {...attrs} 
              >
                <div className={styles.userProfile}>
                  <div className={styles.userInfo}>
                    <img
                      src="https://media.istockphoto.com/id/1443209389/vi/vec-to/s%C3%A1ch-v%E1%BA%BD-ngu%E1%BB%87ch-ngo%E1%BA%A1c.jpg?s=612x612&w=0&k=20&c=a-bJw6Ic1FLS5Ri4TlRqOh0bREGyVO_2W52seuNupuk=" 
                      alt="avatar"
                      className={styles.menuAvatar}
                    />
                    <span className={styles.userEmail}>{user?.email}</span>
                  </div>
                  <button onClick={goToProfile} className={styles.profileButton}>
                    My Profile
                  </button>
                  <button onClick={handleLogout} className={styles.logoutButton}>
                    Đăng xuất
                  </button>
                </div>
              </div>
            )}
          >
            <div
              className={styles.userMenu}
              onClick={() => setVisible(!visible)} 
            >
              <span className={styles.username}>Tài khoản</span>
              <img
                src="https://media.istockphoto.com/id/1443209389/vi/vec-to/s%C3%A1ch-v%E1%BA%BD-ngu%E1%BB%87ch-ngo%E1%BA%A1c.jpg?s=612x612&w=0&k=20&c=a-bJw6Ic1FLS5Ri4TlRqOh0bREGyVO_2W52seuNupuk=" 
                alt="avatar"
                className={styles.avatar}
              />
            </div>
          </HeadlessTippy>
        )}
      </nav>

      {/* Login/Signup Modal */}
      <Modal
        title="Welcome to Harmony Library"
        open={isLoginModalVisible}
        onCancel={handleLoginModalCancel}
        footer={null}
        centered
        className={styles.authModal}
      >
        <div className={styles.modalContent}>
          <p>Please select an option to continue:</p>
          <div className={styles.modalButtons}>
            <button onClick={goToLogin} className={styles.loginModalButton}>
              Login
            </button>
            <button onClick={goToSignup} className={styles.signupModalButton}>
              Create Account
            </button>
          </div>
        </div>
      </Modal>
    </header>
  );
};

export default Header;
